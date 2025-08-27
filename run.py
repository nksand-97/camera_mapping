from dataset import KITTIDataset
from depth_estimator import DepthEstimator
from map_data import MapData
from utils import *


def main():
    config = load_config("config.yaml")

    ds = KITTIDataset(config)
    de = DepthEstimator(config)
    mp = MapData()

    for index_frame in range(len(ds)):
        print("\rCurrent frame number: {}/{}".format(index_frame, len(ds)-1), end="")

        image = ds.get()
        depth_image = de.estimate(image)
        mp.save(depth_image)
    
    # debug
    mp.debug_visualize_depth()


if __name__ == "__main__":
    main()