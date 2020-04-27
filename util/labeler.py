import copy
import pandas as pd

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
        gray_code = self.color_df['gray'].to_numpy()
        label_code = self.color_df['label_value'].to_numpy()

        # Create dictionary from [ gray_code, label_code]
        self.color_to_label_map = dict(zip(gray_code, label_code))

    def label(self, image):

        # Clone the images
        labeled_image = copy.deepcopy(image)

        # Label All as unknown
        labeled_image[:, :] = 0

        # For each color info, label those pixels
        for color in self.color_to_label_map.keys():
            # Replace color
            labeled_image[image == color] = self.color_to_label_map[color]

        # Return the labeled image
        return labeled_image
