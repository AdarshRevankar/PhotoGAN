import copy
import pandas as pd
import numpy as np

"""
Author  : Adarsh Revankar
Date    : 26-04-2020
"""


class Labeler:
    """
    Labeler Does the Labeling job, which intern used by the GAN
    GAN requires a specific way of labeling range from 0 - 183.
    This labeling is done here
    """

    def __init__(self, opt):
        # Prepare a conversion dictionary
        self.color_df = pd.read_csv(opt.color_code_path)
        rgb_code = self.color_df[['red', 'green', 'blue']].to_numpy()
        label_code = self.color_df['label_value'].to_numpy()

        # Create dictionary from [ gray_code, label_code]
        self.color_to_label_map = dict(zip(label_code, rgb_code))

    def label(self, image):
        # Clone the images
        labeled_image = np.zeros(image.shape[:2]).astype('uint8')

        # # Label All as unknown
        # labeled_image[:, :] = 0

        # For each color info, label those pixels
        for label, rgb_lst in self.color_to_label_map.items():
            # Replace color
            labeled_image[
                (image[:, :, 0] == rgb_lst[0]) &
                (image[:, :, 1] == rgb_lst[1]) &
                (image[:, :, 2] == rgb_lst[2])] = label

        # Return the labeled image
        return labeled_image
