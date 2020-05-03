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
        self.drawings_path = opt.drawings_path
        self.files = os.listdir(self.drawings_path)
        self.N = len(self.files)
        self.color_scheme = opt.color
        self.size = (opt.width, opt.height)

    def load(self):
        """
        Loads the Grayscale images from the path
        """
        images = []

        # If RGB then isGrayscale as false
        is_gray = True if self.color_scheme == 'rgb' else False

        for file in self.files:
            # Read the image
            image = io.imread(os.path.join(self.drawings_path, file))[:, :, :3]

            # Resize the image
            images.append((resize(image=image, anti_aliasing=False, output_shape=self.size) * 255.).astype('uint8'))

        return images
