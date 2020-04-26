import os
from skimage import io
from skimage.transform import resize

"""
Author  : Adarsh Revankar
Date    : 26-04-2020
"""


class Loader:
    def __init__(self, opt):
        self.files = []
        self.inpPath = opt.input_path
        self.files = os.listdir(self.inpPath)
        self.N = len(self.files)
        self.color_scheme = opt.color
        self.size = (opt.width, opt.height)
        self.colors_meta_path = opt.colors_meta_path

    def load(self):
        """
        Loads the Grayscale images from the path
        """
        images = []

        # If RGB then isGrayscale as false
        is_gray = True if self.color_scheme == 'rgb' else False

        # Process colors meta
        colors_meta = open(self.colors_meta_path, 'r').readlines()

        image_color_meta = []

        for file in self.files:
            # Process color info
            c_line = [line for line in colors_meta if file in line.split()][0].replace('\n', '')
            image_color_meta.append([int(x) for x in c_line.split(' ')[1:]])

            # Read the image
            image = io.imread(os.path.join(self.inpPath, file), as_gray=is_gray)

            # Resize the image
            images.append((resize(image=image, output_shape=self.size) * 255).astype('uint8'))

        return images, image_color_meta
