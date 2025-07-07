import joblib

def get_motion_len(motion_file_path: str):
    """
    Get the length of the motion from a motion file.
    
    Args:
        motion_file_path (str): Path to the motion file.
        
    Returns:
        int: Length of the motion.
    """
    motion_data = joblib.load(motion_file_path)
    motion_data = motion_data[list(motion_data.keys())[0]]
    fps = motion_data["fps"]
    dt = 1.0 / fps
    num_frames = motion_data["root_rot"].shape[0]
    motion_len = dt * (num_frames - 1)
    return motion_len