import numpy as np

# debug
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


class MapData:
    def __init__(self):
        self.depth_images = []
        self.poses_R = []
        self.poses_t = []

    
    def save(self, depth_image, pose):
        if depth_image:
            self.depth_images.append(depth_image)
        
        self.poses_R.append(pose["R"])
        self.poses_t.append(pose["t"])
    

    # debug
    # フレーム毎の深度画像の描画
    def debug_visualize_depth(self):
        for index, depth in enumerate(self.depth_images):
            plt.figure()
            plt.axis("off")
            plt.imshow(depth)
            plt.savefig("debug/output/depth/{}.png".format(str(index).zfill(4)))
            plt.close()


    # 推定位置の描画    
    def debug_visualize_pose(self):

        poses_t_arr = np.array(self.poses_t)
        plt.xlabel("x [m]")
        plt.ylabel("z [m]")
        # plt.xlim(-300, 300)
        # plt.ylim(-100, 500)
        plt.plot(poses_t_arr[:, 0], poses_t_arr[:, 2], c="black", linestyle="--", label="ground_truth")
        plt.grid(color="lightgray", linestyle="--")
        plt.legend()
        plt.savefig("debug/output/pose.png")
        plt.close()