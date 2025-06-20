from typing import Dict, List, Tuple, Callable, Optional, Any, Union, TypeVar
import numpy as np
from omegaconf import OmegaConf, DictConfig, ListConfig
from loguru import logger
from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from humanoidverse.utils.helpers import parse_observation, np2torch, torch2np
from humanoidverse.utils.motion_lib.motion_lib_robot_WJX import MotionLibRobotWJX as MotionLibRobot

import time
import torch
import os
from pathlib import Path
from description.robots.dtype import RobotExitException
from isaac_utils.rotations import calc_heading_quat_inv, my_quat_rotate, calc_yaw_heading_quat_inv


URCIRobotType = TypeVar('URCIRobotType', bound='URCIRobot')
ObsCfg = Union[DictConfig, Callable[[URCIRobotType],np.ndarray]]
URCIPolicyObs = Tuple[ObsCfg, Callable]

CfgType = Union[OmegaConf, ListConfig, DictConfig]


def wrap_to_pi_float(angles:float):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

class URCIRobot:
    REAL: bool
    BYPASS_ACT: bool
    SWITCH_EMA: bool
    
    dt: float # big dt, not small dt
    clip_observations: float
    cfg: OmegaConf
    
    q: np.ndarray
    dq: np.ndarray
    # pos: np.ndarray
    # vel: np.ndarray
    quat: np.ndarray  # XYZW
    omega: np.ndarray
    gvec: np.ndarray
    rpy: np.ndarray
    
    act: np.ndarray
    
    _obs_cfg_obs: ObsCfg
    # _obs_cfg_obs: Optional[ObsCfg]=None
    _ref_pid = -2 # reference policy id
                    # Normal pid: (0,1,2,3)
                    # Special pid: (-2, -3, -4)
    _pid_size = -1
    
    def __init__(self, cfg: CfgType):
        self.BYPASS_ACT = cfg.deploy.BYPASS_ACT
        self.SWITCH_EMA = cfg.deploy.SWITCH_EMA
        
        self.cfg: OmegaConf = cfg
        self._obs_cfg_obs = cfg.obs
        self.device: str = "cpu"
        self.dt: float = cfg.deploy.ctrl_dt
        self.timer: int = 0
        
        
        self.num_actions = cfg.robot.actions_dim
        self.heading_cmd = cfg.deploy.heading_cmd   
        self.clip_action_limit: float = cfg.robot.control.action_clip_value
        self.clip_observations: float = cfg.env.config.normalization.clip_observations
        self.action_scale: float = cfg.robot.control.action_scale
        
        self._make_init_pose()
        self._make_buffer()
        if cfg.log_task_name == "motion_tracking":
            self.is_motion_tracking = True
            self._make_motionlib([])
        else:
            self.is_motion_tracking = False
        if 'save_motion' in cfg.env.config:
            self.save_motion = bool(cfg.env.config.save_motion)
            if self.save_motion: self._make_init_save_motion()
        else: 
            self.save_motion = False
        
    
    def routing(self, cfg_policies: List[URCIPolicyObs]):
        """
            Usage: Input a list of Policy, and the robot can switch between them.
            
            - Policies are indexed by integers (Pid), 0 to len(cfg_policies)-1.
            - special pid: 
                - -2: Reset the robot.
                - 0: Default policy, should be stationary. 
                    - The Robot will switch to this policy once the motion tracking is Done or being Reset.
            - Switching Mechanism:
                - The instance (MuJoCo or Real) should implement the Pid control logic. It can be changed at any time.
                - When the instance want to Reset the robot, it should set the pid to -2.
        """
        self._pid_size = len(cfg_policies)
        self._make_motionlib(cfg_policies)
        self._check_init()
        self.cmd[3]=self.rpy[2]
        cur_pid = -1

        try: 
            while True:
                t1 = time.time()
                
                if cur_pid != self._ref_pid or self._ref_pid == -2:
                    if self._ref_pid == -2:
                        self.Reset()
                        self._ref_pid = 0
                        t1 = time.time()
                        ...
                    
                    
                    self._ref_pid %= self._pid_size
                    assert self._ref_pid >= 0 and self._ref_pid < self._pid_size, f"Invalid policy id: {self._ref_pid}"
                    self.TrySaveMotionFile(pid=cur_pid)       
                    logger.info(f"Switch to the policy {self._ref_pid}")

                    
                    cur_pid = self._ref_pid
                    self.SetObsCfg(cfg_policies[cur_pid][0])
                    policy_fn = cfg_policies[cur_pid][1]
                    if self.SWITCH_EMA:
                        self.old_act = self.act.copy()
                    # print('Debug: ',self.Obs()['actor_obs'])
                    # breakpoint()

                    
                    # breakpoint()
                
                self.UpdateObs()
                
                action = policy_fn(self.Obs())[0]
                
                if self.BYPASS_ACT: action = np.zeros_like(action)
                
                if self.SWITCH_EMA and self.timer <10:
                    self.old_act = self.old_act * 0.9 + action * 0.1
                    action = self.old_act
                    
                
                self.ApplyAction(action)
                
                
                self.TrySaveMotionStep()
                
                if self.motion_len > 0 and self.ref_motion_phase > 1.0:
                    # self.Reset()
                    if self._ref_pid == 0:
                        self._ref_pid = -2
                    else:
                        self._ref_pid = 0
                    self.TrySaveMotionFile(pid=cur_pid)
                    logger.info("Motion End. Switch to the Default Policy")
                t2 = time.time()
                
                # print(f"t2-t1 = {(t2-t1)*1e3} ms")
                if self.REAL:
                # if True:
                    # print(f"t2-t1 = {(t2-t1)*1e3} ms")
                    remain_dt = self.dt - (t2-t1)
                    if remain_dt > 0:
                        time.sleep(remain_dt)
                    else:
                        logger.warning(f"Warning! delay = {t2-t1} longer than policy_dt = {self.dt} , skip sleeping")
        except RobotExitException as e:
            self.TrySaveMotionFile(pid=cur_pid)
            raise e
        ...
    
    def _reset(self):
        raise NotImplementedError("Not implemented")
    
    def Reset(self):
        # self.TrySaveMotionFile()
        
        self._reset()
        
        self.act[:] = 0
        self.history_handler.reset([0])
        self.timer: int = 0
        self.cmd: np.ndarray = np.array(self.cfg.deploy.defcmd)
        self.cmd[3]=self.rpy[2]
        
        
        self.UpdateObs()
        self.motion_time = 0.0
        self.ref_init_yaw[0] = self.rpy[2]
        
    def _apply_action(self, target_q):
        raise NotImplementedError("Not implemented")
    
    # @_prof_applyaction
    def ApplyAction(self, action): 
        self.timer += 1
        
        self.act = action.copy()
        target_q = np.clip(action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
        
        self._apply_action(target_q)
        

    
    def Obs(self)->Dict[str, np.ndarray]:
        return {'actor_obs': torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)}
    
    
    def _get_state(self):
        raise NotImplementedError("Not implemented")
    
    def GetState(self):
        self._get_state()
        
        
        if self.heading_cmd:
            self.cmd[2] = np.clip(0.5*wrap_to_pi_float(self.cmd[3] - self.rpy[2]), -1., 1.)
            
        if self.is_motion_tracking:
            self.motion_time = (self.timer) * self.dt 
            self.ref_motion_phase = self.motion_time / self.motion_len
        print("cmd: ", self.cmd,end='\r\b')
    

    def SetObsCfg(self, obs_cfg: ObsCfg):
        if isinstance(obs_cfg, DictConfig):
            self._obs_cfg_obs = obs_cfg
            self.motion_len = self._obs_cfg_obs.motion_len
        elif isinstance(obs_cfg, Callable):
            self._obs_cfg_obs = obs_cfg
            self.motion_len = -1
        else:
            raise NotImplementedError("Not implemented")
        
        # self.act[:] = 0
        # self.history_handler.reset([0])
        self.motion_lib = self.motion_libs[self._ref_pid]
        self.timer: int = 0
        self.cmd: np.ndarray = np.array(self.cfg.deploy.defcmd)
        self.cmd[3]=self.rpy[2]
        self.ref_init_yaw[0] = self.rpy[2]
        if 'ref_motion_phase' in self.history_handler.history.keys():
            self.ref_motion_phase = 0.
            self.history_handler.history['ref_motion_phase']*=0
        self.KickMotionLib()
        # self.UpdateObsWoHistory() # Recompute obs with new _obs_cfg_obs
        ...
    
    def UpdateObsWoHistory(self):
        if isinstance(self._obs_cfg_obs, Callable):
            # self.obs_buf_dict = dict()
            self.obs_buf_dict['actor_obs'] = self._obs_cfg_obs(self)
            return 
        
        obs_cfg_obs = self._obs_cfg_obs
        
        self.obs_buf_dict_raw = {}
        
        noise_extra_scale = 0.
        for obs_key, obs_config in obs_cfg_obs.obs_dict.items():
            if not obs_key=='actor_obs': continue
            self.obs_buf_dict_raw[obs_key] = dict()

            parse_observation(self, obs_config, self.obs_buf_dict_raw[obs_key], obs_cfg_obs.obs_scales, obs_cfg_obs.noise_scales, noise_extra_scale)
        
        self.obs_buf_dict = dict()
        
        for obs_key, obs_config in obs_cfg_obs.obs_dict.items():
            if not obs_key=='actor_obs': continue
            obs_keys = sorted(obs_config)
            # (Pdb) sorted(obs_config)
            # ['actions', 'base_ang_vel', 'base_lin_vel', 'dif_local_rigid_body_pos', 'dof_pos', 'dof_vel', 'dr_base_com', 'dr_ctrl_delay', 'dr_friction', 'dr_kd', 'dr_kp', 'dr_link_mass', 'history_critic', 'local_ref_rigid_body_pos', 'projected_gravity', 'ref_motion_phase']
            
            # print("obs_keys", obs_keys, self.obs_buf_dict_raw[obs_key])            
            # print("obs_keys:", obs_keys)
            # print("obs shape:", {key: self.obs_buf_dict_raw[obs_key][key].shape for key in obs_keys})            
            self.obs_buf_dict[obs_key] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1)
            
            
        clip_obs = self.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            if not obs_key=='actor_obs': continue
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)
        # breakpoint()

    def UpdateObsForHistory(self):
        hist_cfg_obs = self.cfg.obs
        
        self.hist_obs_dict = {}
        
        noise_extra_scale = 0.
        # Compute history observations
        history_obs_list = self.history_handler.history.keys()
        # print(f"history_obs_list: {history_obs_list}")
        parse_observation(self, history_obs_list, self.hist_obs_dict, hist_cfg_obs.obs_scales, hist_cfg_obs.noise_scales, noise_extra_scale)
        
        for key in self.history_handler.history.keys():
            self.history_handler.add(key, self.hist_obs_dict[key])
    
    _kick_motion_res_counter = -1
    _kick_motion_res_buffer: Optional[Dict[str, torch.Tensor]] = None
    def _kick_motion_res(self)->Dict[str, torch.Tensor]:
        if self._kick_motion_res_counter == self.timer:
            return self._kick_motion_res_buffer # type: ignore
        
        self._kick_motion_res_counter = self.timer
        
        t0 = max(self.timer - 1, 0) * self.dt
        t1 = self.timer * self.dt
        motion_times = torch.tensor([t0, t1], dtype=torch.float32, device='cpu')
        motion_ids = torch.zeros(2, dtype=torch.long, device='cpu')
        seq = self.motion_lib.get_motion_state(motion_ids, motion_times)
        
        self._kick_motion_res_buffer = {k: v[-1:].contiguous() for k, v in seq.items()}
        return self._kick_motion_res_buffer
    
    
    # @_prof_getmotion
    def KickMotionLib(self):
        # motion_time x motion lib -> ref state  for obs
        if self.motion_lib is None:
            return
        motion_res =  self._kick_motion_res()
            # (Pdb) motion_res.keys()
            # dict_keys(['contact_mask', 'root_pos', 'root_rot', 'dof_pos', 'root_vel', 'root_ang_vel', 'dof_vel', 'motion_aa', 'motion_bodies', 'rg_pos', 'rb_rot', 'body_vel', 'body_ang_vel', 'rg_pos_t', 'rg_rot_t', 'body_vel_t', 'body_ang_vel_t'])

        current_yaw = self.rpy[2]
        self.relyaw = current_yaw - self.ref_init_yaw
        
        relyaw_heading_inv_quat = calc_yaw_heading_quat_inv(torch.from_numpy(self.relyaw).to(dtype=torch.float32).unsqueeze(0))
        relyaw_heading_inv_quat_expand = relyaw_heading_inv_quat.unsqueeze(1).expand(-1, 27, -1).reshape(-1, 4)

        heading_inv_rot = calc_heading_quat_inv(torch.from_numpy(self.quat).to(dtype=torch.float32).unsqueeze(0), w_last=True) #xyzw
        # # expand to (B*num_rigid_bodies, 4) for fatser computation in jit
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, 27, -1).reshape(-1, 4)


        ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        ref_joint_vel = motion_res["dof_vel"] # [num_envs, num_dofs]
        ref_body_vel_extend = motion_res["body_vel_t"] # [num_envs, num_markers, 3]
        
        
        global_ref_body_vel = ref_body_vel_extend.view(1, -1, 3)
        if heading_inv_rot_expand.shape[0] == 0:
            local_ref_rigid_body_vel_flat = torch.zeros_like(global_ref_body_vel.view(-1, 3))
        else:
            local_ref_rigid_body_vel_flat = my_quat_rotate(
                heading_inv_rot_expand.view(-1, 4),
                global_ref_body_vel.view(-1, 3)
            )
        # local_ref_rigid_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_body_vel.view(-1, 3))

        ## diff compute - kinematic joint position
        self.dif_joint_angles = (ref_joint_pos - self.q).to(dtype=torch.float32).view(-1)
        ## diff compute - kinematic joint velocity
        self.dif_joint_velocities = (ref_joint_vel - self.dq).to(dtype=torch.float32).view(-1)      
        self._obs_global_ref_body_vel = global_ref_body_vel.view(-1) # (num_envs, num_rigid_bodies*3)
        self._obs_local_ref_rigid_body_vel = local_ref_rigid_body_vel_flat.view(-1) # (num_envs, num_rigid_bodies*3)
        self._obs_local_ref_rigid_body_pos_relyaw = my_quat_rotate(relyaw_heading_inv_quat_expand.view(-1, 4), 
                                                                   global_ref_body_vel.view(-1, 3)).view(-1)
        
        # print("motion_res['root_rot']", motion_res['root_rot']);        breakpoint()
        return

    def UpdateObs(self):
        self.GetState()
        self.KickMotionLib()
        self.UpdateObsWoHistory()
        self.UpdateObsForHistory()
        
    def _check_init(self):
        assert self.dt is not None, "dt is not set"
        assert self.dt>0 and self.dt < 0.1, "dt is not in the valid range"
        assert self.cfg is not None or not isinstance(self.cfg, OmegaConf), "cfg is not set"
        
        assert self.num_dofs is not None, "num_dofs is not set"
        assert self.num_dofs == 23, "In policy level, only 23 dofs are supported for now"
        assert self.kp is not None and type(self.kp) == np.ndarray and self.kp.shape == (self.num_dofs,), "kp is not set"
        assert self.kd is not None and type(self.kd) == np.ndarray and self.kd.shape == (self.num_dofs,), "kd is not set"
        
        assert (self.dof_init_pose is not None and type(self.dof_init_pose) == np.ndarray and 
                    self.dof_init_pose.shape == (self.num_dofs,)), "dof_init_pose is not set"
        
        assert self.tau_limit is not None and type(self.tau_limit) == np.ndarray and self.tau_limit.shape == (self.num_dofs,), "tau_limit is not set"
        
        assert self.BYPASS_ACT is not None, "BYPASS_ACT is not set"
        assert self.BYPASS_ACT in [True, False], "BYPASS_ACT is not a boolean, got {self.BYPASS_ACT}"
        
        assert self._pid_size > 0, "pid_size is not correctly set"
    
    def _make_init_pose(self):
        cfg_init_state = self.cfg.robot.init_state
        self.body_names = self.cfg.robot.body_names
        self.dof_names = self.cfg.robot.dof_names
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        assert self.num_dofs == 23, "Only 23 dofs are supported for now"
        
        
        dof_init_pose = cfg_init_state.default_joint_angles
        dof_effort_limit_list = self.cfg.robot.dof_effort_limit_list
        
        self.dof_init_pose = np.array([dof_init_pose[name] for name in self.dof_names])
        self.tau_limit = np.array(dof_effort_limit_list)
        
        
        self.kp = np.zeros(self.num_dofs)
        self.kd = np.zeros(self.num_dofs)
        
        
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.robot.control.stiffness.keys():
                if dof_name in name:
                    self.kp[i] = self.cfg.robot.control.stiffness[dof_name]
                    self.kd[i] = self.cfg.robot.control.damping[dof_name]
                    found = True
                    logger.debug(f"PD gain of joint {name} were defined, setting them to {self.kp[i]} and {self.kd[i]}")
            if not found:
                raise ValueError(f"PD gain of joint {name} were not defined. Should be defined in the yaml file.")
        
    def _make_buffer(self):
        self.cmd: np.ndarray = np.array(self.cfg.deploy.defcmd)
        
        self.q = np.zeros(self.num_dofs)
        self.dq = np.zeros(self.num_dofs)
        self.quat = np.zeros(4)  # XYZW
        self.omega = np.zeros(3)
        self.gvec = np.zeros(3)
        self.rpy = np.zeros(3)
        
        self.act = np.zeros(self.num_dofs)
        
        
        self.history_handler = HistoryHandler(1, self.cfg.obs.obs_auxiliary, self.cfg.obs.obs_dims, self.device)
        
        self.motion_lib = None
        self.ref_init_yaw = np.zeros(1,dtype=np.float32)
        self.relyaw = np.zeros(1,dtype=np.float32)
        self.dif_joint_angles = torch.zeros(self.num_dofs, dtype=torch.float32)
        self.dif_joint_velocities = torch.zeros(self.num_dofs, dtype=torch.float32)
        self._obs_global_ref_body_vel = torch.zeros(27*3, dtype=torch.float32)  # 27 rigid bodies, each has 3 velocity components
        self._obs_local_ref_rigid_body_vel = torch.zeros(27*3, dtype=torch.float32)
        self._obs_local_ref_rigid_body_pos_relyaw = torch.zeros(27*3, dtype=torch.float32)
        ...
        
    def _make_motionlib(self, cfg_policies: List[URCIPolicyObs]):
        self.motion_len = self.cfg.obs.motion_len # an initial value
        
        # For all motion tracking policy, load the motion lib file
        self.motion_libs: List[Optional[MotionLibRobot]] = []
        
        for cfg_policy in cfg_policies:
            obs_cfg, policy_fn = cfg_policy
            if isinstance(obs_cfg, DictConfig):
                m_cfg = DictConfig({
                    'motion_file': obs_cfg.motion_file,
                    'asset': self.cfg.robot.motion.asset,
                    'extend_config': self.cfg.robot.motion.extend_config,
                })
                
                motion_lib = MotionLibRobot(m_cfg, num_envs=1, device='cpu')
                motion_lib.load_motions(random_sample=False)[0]
                self.motion_libs.append(motion_lib)
            elif isinstance(obs_cfg, Callable):
                # self.motion_len = -1
                # pass
                self.motion_libs.append(None)
            else:
                raise ValueError(f"Invalid obs_cfg: {obs_cfg}")
            
            
        assert len(self.motion_libs) == len(cfg_policies), f"{len(self.motion_libs)=} != {len(cfg_policies)=}"
        # breakpoint()
        
        
    # ---- Save Motion System----
        
    def _make_init_save_motion(self):
        save_dir = self.cfg.checkpoint.parent.parent / "motions"
        # os.makedirs(save_dir, exist_ok = True)

        self._save_motion_id = 0
        
        if hasattr(self.cfg, 'dump_motion_name'):
            raise NotImplementedError
            self.save_motion_dir = save_dir / (str(self.cfg.eval_timestamp) + "_" + self.cfg.dump_motion_name)
        else:
            self.save_motion_dir = save_dir / f"{self.cfg.env.config.save_note}_URCI_{type(self).__name__}_{self.cfg.eval_timestamp}"
        self.save_motion_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Save Motion Dir: ", self.save_motion_dir)
        OmegaConf.save(self.cfg, self.save_motion_dir / "config.yaml")
        

        self._dof_axis = np.load('humanoidverse/utils/motion_lib/dof_axis.npy', allow_pickle=True)
        self._dof_axis = self._dof_axis.astype(np.float32)

        self.num_augment_joint = len(self.cfg.robot.motion.extend_config)
        self.motions_for_saving: Dict[str, List[np.ndarray]] = {'root_trans_offset':[], 'pose_aa':[], 'dof':[], 'root_rot':[], 'actor_obs':[], 'action':[], 'terminate':[],
                                    'root_lin_vel':[], 'root_ang_vel':[], 'dof_vel':[]}
        self.motion_times_buf = []

        # breakpoint()

    def _get_motion_to_save(self)->Tuple[float, Dict[str, np.ndarray]]:
        raise NotImplementedError("Not implemented")
    
    def TrySaveMotionStep(self):
        if self.save_motion:
            motion_time, motion_data = self._get_motion_to_save()
            self.motion_times_buf.append(motion_time)
            for key, value in motion_data.items():
                if not key in self.motions_for_saving.keys():
                    self.motions_for_saving[key] = []
                    assert isinstance(value, np.ndarray), f"key: {key} is not a np.ndarray"
                self.motions_for_saving[key].append(value)
    
    def TrySaveMotionFile(self, pid: int):
        if self.save_motion:
            import joblib
            from termcolor import colored
            if self.motions_for_saving['root_trans_offset'] == []:
                return
            
            for k, v in self.motions_for_saving.items():
                self.motions_for_saving[k] = np.stack(v).astype(np.float32) # type: ignore
            
            self.motions_for_saving['motion_times'] = np.array(self.motion_times_buf, dtype=np.float32) # type: ignore
            
            dump_data = {}
            keys_to_save = self.motions_for_saving.keys()

            motion_key = f"motion{self._save_motion_id}"
            dump_data[motion_key] = {
                key: self.motions_for_saving[key] for key in keys_to_save
            }
            dump_data[motion_key]['fps'] = 1 / self.dt

            save_path = f'{self.save_motion_dir}/{self._save_motion_id}_pid{pid}_frame{len(self.motions_for_saving["dof"])}_{time.strftime("%Y%m%d_%H%M%S")}.pkl'
            joblib.dump(dump_data, save_path)
            
            logger.info(colored(f"Saved motion data to {save_path}", 'green'))

            self._save_motion_id += 1            
            self.motions_for_saving = {'root_trans_offset':[], 'pose_aa':[], 'dof':[], 'root_rot':[], 'actor_obs':[], 'action':[], 'terminate':[],
                                        'root_lin_vel':[], 'root_ang_vel':[], 'dof_vel':[]}
            self.motion_times_buf = []
        ...
    
    # --------------------------------------------
    
    
    ######################### Observations #########################
    def _get_obs_command_lin_vel(self):
        return np2torch(self.cmd[:2])
    
    def _get_obs_command_ang_vel(self):
        return np2torch(self.cmd[2:3])
    
    def _get_obs_actions(self,):
        return np2torch(self.act)
    
    def _get_obs_base_pos_z(self,):
        # raise NotImplementedError("Not Implemented")
        return np2torch(self.pos[2:3])
    
    def _get_obs_feet_contact_force(self,):
        raise NotImplementedError("Not implemented")
        return self.data.contact.force[:, :].view(self.num_envs, -1)
          
    def _get_obs_base_lin_vel(self,):
        # raise NotImplementedError("Not Implemented")
        # return np2torch(self.vel)
        # print("Warning! base_lin_vel is not allowed to access")
        return torch.zeros(3)
    
    def _get_obs_base_ang_vel(self,):
        return np2torch(self.omega)
    
    def _get_obs_projected_gravity(self,):
        return np2torch(self.gvec)
    
    def _get_obs_dof_pos(self,):
        return np2torch(self.q - self.dof_init_pose)
    
    def _get_obs_dof_vel(self,):
        # print(f"dof_vel: mean:{self.dq.mean()}, std:{self.dq.std()}")
        return np2torch(self.dq)
    
    
    def _get_obs_base_ang_vel_noise(self,):
        return np2torch(self.omega)
    
    def _get_obs_projected_gravity_noise(self,):
        return np2torch(self.gvec)
    
    def _get_obs_dof_pos_noise(self,):
        return np2torch(self.q - self.dof_init_pose)
    
    def _get_obs_dof_vel_noise(self,):
        return np2torch(self.dq)
    
    
    def _get_obs_ref_motion_phase(self):
        # logger.info(f"Phase: {self.ref_motion_phase} | {self.motion_len}")
        return torch.tensor(self.ref_motion_phase).reshape(1,)
    
    def _get_obs_relyaw(self):
        return np2torch(self.relyaw)
    
    def _get_obs_dif_joint_angles(self):
        # print(f"dif_joint_angles: mean:{self.dif_joint_angles.mean()}, std:{self.dif_joint_angles.std()}")
        return self.dif_joint_angles

    def _get_obs_dif_joint_velocities(self):
        # print(f"dif_joint_velocities: mean:{self.dif_joint_velocities.mean()}, std:{self.dif_joint_velocities.std()}")
        return self.dif_joint_velocities
    
    def _get_obs_global_ref_rigid_body_vel(self):
        # print(f"global_ref_body_vel: mean:{self._obs_global_ref_body_vel.mean()}, std:{self._obs_global_ref_body_vel.std()}")
        return self._obs_global_ref_body_vel
    
    def _get_obs_local_ref_rigid_body_vel(self):
        # print(f"local_ref_rigid_body_vel: mean:{self._obs_local_ref_rigid_body_vel.mean()}, std:{self._obs_local_ref_rigid_body_vel.std()}")
        return self._obs_local_ref_rigid_body_vel
    
    
    def _get_obs_local_ref_rigid_body_pos_relyaw(self):
        return self._obs_local_ref_rigid_body_pos_relyaw
    
    def _get_obs_dif_local_rigid_body_pos(self):
        raise NotImplementedError("Not implemented")
        return self._obs_dif_local_rigid_body_pos
    
    def _get_obs_local_ref_rigid_body_pos(self):
        raise NotImplementedError("Not implemented")
        return self._obs_local_ref_rigid_body_pos
    
    def _get_obs_vr_3point_pos(self):
        raise NotImplementedError("Not implemented")
        return self._obs_vr_3point_pos
    
    def _get_obs_history(self,):
        obs_cfg_obs = self._obs_cfg_obs
        assert "history" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_short_history(self,):
        obs_cfg_obs = self._obs_cfg_obs
        assert "short_history" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['short_history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_long_history(self,):
        obs_cfg_obs = self._obs_cfg_obs
        assert "long_history" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['long_history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_history_actor(self,):
        obs_cfg_obs = self._obs_cfg_obs
        assert "history_actor" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
            
        # print("history_tensors:", {key: history_tensors[i].shape for i, key in enumerate(sorted(history_config.keys()))})
        # breakpoint()
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_history_critic(self,):
        obs_cfg_obs = self._obs_cfg_obs
        assert "history_critic" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['history_critic']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
