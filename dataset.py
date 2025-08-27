import os
import glob
import cv2


class KITTIDataset:
    def __init__(self, config):
        self.curr_idx = 0

        dataset_path = config["kitti"]["dataset_path"]
        img_path = os.path.join(dataset_path, "sequences/00/image_2")
        len_seq = len(glob.glob(os.path.join(img_path, "*.png")))
        self.img_path_list = []
        for i in range(len_seq):
            self.img_path_list.append(os.path.join(img_path, "00{}.png".format(str(i).zfill(4)))) 


    def __len__(self):
        return len(self.img_path_list)


    def get(self):
        image = cv2.imread(self.img_path_list[self.curr_idx])
        
        self.curr_idx += 1

        return image