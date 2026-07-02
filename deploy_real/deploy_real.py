from typing import Union
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import (
        create_damping_cmd, 
        create_zero_cmd, 
        init_cmd_hg, 
        init_cmd_go,  
        MotorMode
    )
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from common.motion_lib_helper import get_motion_len
from config import Config
from collections import deque
import onnxruntime as ort


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.frame_stack = self.config.frame_stack
        
        self.motion_file = self.config.motion_file
        self.motion_len = get_motion_len(self.motion_file)

        self.remote_controller = RemoteController()

        # Initialize the policy network
        # self.policy = torch.jit.load(config.policy_path)
        self.policy = ort.InferenceSession(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.counter = 0

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.start_time = time.time()

        # Initialize histories for each observation type
        self.history = {
            "action": deque(maxlen=self.frame_stack-1),
            "omega": deque(maxlen=self.frame_stack-1),
            "qj": deque(maxlen=self.frame_stack-1),
            "dqj": deque(maxlen=self.frame_stack-1),
            "gravity_orientation": deque(maxlen=self.frame_stack-1),
            "ref_motion_phase": deque(maxlen=self.frame_stack-1),
        }

        for _ in range(self.frame_stack - 1):
            for key in self.history:
                if key in ["action", "qj", "dqj"]:
                    self.history[key].append(torch.zeros(1, self.config.num_actions, dtype=torch.float))
                elif key in ["omega", "gravity_orientation"]:
                    self.history[key].append(torch.zeros(1, 3, dtype=torch.float))
                elif key == "ref_motion_phase":
                    self.history[key].append(torch.zeros(1, 1, dtype=torch.float))
                else:
                    raise ValueError(f"Not Implement: {key}")

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 5
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        self.default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        self.default_angles = self.default_pos[0:23]
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = self.default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
    

    def run(self):

        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        
        # 1. Gather individual history tensors
        action_hist_tensor = torch.cat([self.history["action"][i] for i in range(self.frame_stack-1)], dim=1)
        omega_hist_tensor = torch.cat([self.history["omega"][i] for i in range(self.frame_stack-1)], dim=1)
        qj_hist_tensor = torch.cat([self.history["qj"][i] for i in range(self.frame_stack-1)], dim=1)
        dqj_hist_tensor = torch.cat([self.history["dqj"][i] for i in range(self.frame_stack-1)], dim=1)
        gravity_orientation_hist_tensor = torch.cat([self.history["gravity_orientation"][i] for i in range(self.frame_stack-1)], dim=1)
        ref_motion_phase_hist_tensor = torch.cat([self.history["ref_motion_phase"][i] for i in range(self.frame_stack-1)], dim=1)
        
        # 2. Concatenate all parts into a single observation tensor
        obs_hist = torch.cat([
            action_hist_tensor,
            omega_hist_tensor,
            qj_hist_tensor,
            dqj_hist_tensor,
            gravity_orientation_hist_tensor,
            ref_motion_phase_hist_tensor
        ], dim=1)

        # 3. Get the current observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        ref_motion_phase = ((self.counter * self.config.control_dt) % self.motion_len) / self.motion_len

        num_actions = self.config.num_actions

        curr_obs = np.zeros(self.config.num_obs, dtype=np.float32)
        curr_obs[: num_actions] = self.action
        curr_obs[num_actions: num_actions + 3] = ang_vel
        curr_obs[num_actions + 3: 2 * num_actions + 3] = qj_obs
        curr_obs[2 * num_actions + 3: 3 * num_actions + 3] = dqj_obs
        curr_obs[3 * num_actions + 3: 3 * num_actions + 6] = gravity_orientation
        curr_obs[6 + 3 * num_actions] = ref_motion_phase

        curr_obs_tensor = torch.from_numpy(curr_obs).unsqueeze(0)
        
        # 4. Get obs buffer, the order is key's alphabetical order
        self.obs_buf = torch.cat([
            curr_obs_tensor[:, :3 * num_actions + 3], 
            obs_hist, 
            curr_obs_tensor[:, 3 * num_actions + 3:]], 
            dim=1
        )

        # 5. Update the history
        self.history["action"].appendleft(curr_obs_tensor[:, :num_actions])
        self.history["omega"].appendleft(curr_obs_tensor[:, num_actions:num_actions+3])
        self.history["qj"].appendleft(curr_obs_tensor[:, num_actions+3:num_actions+3+num_actions])
        self.history["dqj"].appendleft(curr_obs_tensor[:, num_actions+3+num_actions:num_actions+3+2*num_actions])
        self.history["gravity_orientation"].appendleft(curr_obs_tensor[:, num_actions+3+2*num_actions:num_actions+3+2*num_actions+3])
        self.history["ref_motion_phase"].appendleft(curr_obs_tensor[:, -1].unsqueeze(0))
        
        # 6. Get policy's inference
        input_name = self.policy.get_inputs()[0].name
        outputs = self.policy.run(None, {input_name: self.obs_buf.numpy()})
        self.action = outputs[0].squeeze()
        target_dof_pos = self.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
