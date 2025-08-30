import yaml
import numpy as np


def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config


def convert_pose_rt(pose):
    pose = np.array(pose).reshape(3, 4)
    R = pose[:, :3]
    t = pose[:, 3]

    return R, t