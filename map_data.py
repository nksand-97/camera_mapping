# debug
import matplotlib.pyplot as plt


class MapData:
    def __init__(self):
        self.depth_images = []

    
    def save(self, depth_image):
        self.depth_images.append(depth_image)
    

    # debug
    def debug_visualize_depth(self):
        for index, depth in enumerate(self.depth_images):
            plt.figure()
            plt.axis("off")
            plt.imshow(depth)
            plt.savefig("debug/output/depth/{}.png".format(str(index).zfill(4)))
            plt.close()