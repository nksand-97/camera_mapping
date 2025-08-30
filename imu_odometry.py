import numpy as np
from scipy.spatial.transform import Rotation


class IMUOdometry:
    def __init__(self, config):
        self.calib_r = config["kitti"]["imu_calib_r"]

    
    def localization(self, imu_data, pose):
        vx, vy, vz = imu_data["vx"], imu_data["vy"], imu_data["vz"]
        wx, wy, wz = imu_data["wx"], imu_data["wy"], imu_data["wz"]

        angle = Rotation.from_matrix(pose["R"]).as_euler("zyx")
        delta_angle = np.array([-wy, -wz, wx]) * 0.1
        angle = angle + delta_angle
        pose["R"] = Rotation.from_euler("zyx", angle).as_matrix()

        x = pose["t"]
        print(x)
        delta_x = np.array([-vy, -vz, vx]) * 0.1
        pose["t"] = x + pose["R"] @ delta_x

        return pose