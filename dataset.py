import os
import glob
import cv2

from utils import *


# KITTI Datasetの計測データを管理する
class KITTIDataset:
    def __init__(self, config):
        self.curr_idx = 0
        dataset_path = config["kitti"]["dataset_path"]

        # 画像データのパスのリストを取得
        # 画像データ自体はget関数が呼び出されたタイミングで1枚ずつ読込む
        img_path = os.path.join(dataset_path, "sequences/00/image_2")
        len_seq = len(glob.glob(os.path.join(img_path, "*.png")))
        self.img_path_list = []
        for i in range(len_seq):
            self.img_path_list.append(os.path.join(img_path, "00{}.png".format(str(i).zfill(4))))

        # GNSSデータのリストを取得
        gnss_path = os.path.join(dataset_path, "poses/00.txt")
        with open(gnss_path, "r") as f:
            gnss_data_all = f.read()
        self.poses_gnss = []
        gnss_data_seq = gnss_data_all.split("\n")
        for frame in gnss_data_seq:
            R, t = convert_pose_rt([float(data) for data in frame.split(" ")])
            self.poses_gnss.append({"R": R, "t": t})
        
        # IMUデータのリストを取得
        imu_path = os.path.join(dataset_path, "imu_raw/data")
        self.imu_data_list = []
        for i in range(len_seq):
            with open(os.path.join(imu_path, "{}.txt".format(str(i).zfill(10))), "r") as f:
                imu_data_row = f.read()[:-1].split(" ")
                vx = float(imu_data_row[8])
                vy = float(imu_data_row[9])
                vz = float(imu_data_row[10])
                wx = float(imu_data_row[20])
                wy = float(imu_data_row[21])
                wz = float(imu_data_row[22])
                self.imu_data_list.append({
                    "vx": vx, "vy": vy, "vz": vz, "wx": wx, "wy": wy, "wz": wz
                })


    def __len__(self):
        return len(self.img_path_list)


    # 呼び出すごとに次の計測データを読込む
    def get(self):
        # 画像データ
        image = cv2.imread(self.img_path_list[self.curr_idx])

        # GNSSデータ
        pose_gnss = self.poses_gnss[self.curr_idx]

        # IMUデータ
        imu_data = self.imu_data_list[self.curr_idx]
        
        self.curr_idx += 1

        return image, pose_gnss, imu_data