from dataset import KITTIDataset
from depth_estimator import DepthEstimator
from map_data import MapData
from imu_odometry import IMUOdometry
from extended_kalman_filter import ExtendedKalmanFilter
from utils import *


def main():
    config = load_config("config.yaml")

    ds = KITTIDataset(config)
    de = DepthEstimator(config)
    mp = MapData()
    imo = IMUOdometry(config)
    ekf = ExtendedKalmanFilter()

    init_pose_state = True

    for index_frame in range(len(ds)):
        depth_image = None

        print("\rCurrent frame number: {}/{}".format(index_frame, len(ds)-1), end="")

        # センサデータの取得
        image, pose_gnss, imu_data = ds.get()

        # 初期位置はGNSS情報を使用する
        if init_pose_state:
            pose_est = pose_gnss
            init_pose_state = False

        else:
            # IMU計測データから位置推定
            pose_imu = imo.localization(imu_data, pose_est)
            
            # 深度推定
            use_depth_image = config["app"]["use_depth_image"]
            if use_depth_image:
                depth_image = de.estimate(image)

            # 位置推定結果を統合させ、出力する自己位置を計算
            # pose_est = ekf.integrate(pose_gnss, pose_imu, pose_vio)
            pose_est = ekf.integrate(pose_gnss, pose_imu)
        
        # 地図データ保存
        mp.save(depth_image, pose_est)
    
    # debug
    mp.debug_visualize_depth()
    mp.debug_visualize_pose()


if __name__ == "__main__":
    main()