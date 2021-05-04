from typing import Dict, Tuple, List
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


class SegmenterInterface(object):
    def enhance_image(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Enhance the image so that you can more easily threshold it, and create a grayscale image too.

        :param bgr_image: input image in BGR format
        :return: enhanced image in BGRK (K for grayscale) format
        """
        pass

    def threshold_enhanced_image(self, enhanced_bgrk_image: np.ndarray) -> np.ndarray:
        """
        Threshold the enhanced image using the self.THRESHOLD BGRK constant

        :param enhanced_bgrk_image: input image in BGRK format
        :return: binary image for each channel BGRK
        """
        pass

    def clean_thresholded_image(self, thresholded_bgrk_image: np.ndarray) -> np.ndarray:
        """
        Uses morphological operations to clean up thresholded images

        :param thresholded_bgrk_image: input binary image for each channel (BGRK)
        :return: cleaned binary image for each channel (BGRK)
        """
        pass

    def get_combined_thresholded_image(self, cleaned_bgrk_image: np.ndarray) -> np.ndarray:
        """
        Combines the binary image from each channel (BGRK) to form a single binary mask.

        :param cleaned_bgrk_image: cleaned binary image for each channel (BGRK)
        :return: single binary mask to pass to cv2.findContours() and cv2.boundingRect()
        """
        pass

    def get_bounding_boxes(self, input_bgr_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Get bounding boxes (x_left, y_top, width, height) of each object found.

        :param input_bgr_image: input image in BGR format
        :return: list of (x_left, y_top, width, height) bounding boxes
        """
        pass


class PsyduckSegmenter(SegmenterInterface):
    """
    Segment Psyducks within a given BGR image
    """
    def __init__(self):
        self.THRESHOLD = (10, 10, 10, 10)  # TODO: replace with the BGRK values you're looking for


class PikachuSegmenter(SegmenterInterface):
    """
    Segment Pikachus within a given BGR image
    """
    def __init__(self):
        self.THRESHOLD = (10, 10, 10, 10)  # TODO: replace with the BGRK values you're looking for


def main():
    """
    Demo of how to use the argparse module in Python.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', help="path to image file to segment", type=str)
    parser.add_argument('pokemon_type', help="'pikachu' or 'psyduck'", type=str)
    args = parser.parse_args()
    image = cv2.imread(args.input_image)
    if args.pokemon_type == 'pikachu':
        segmenter = PikachuSegmenter()
        print(segmenter.get_bounding_boxes(image))
    elif args.pokemon_type == 'psyduck':
        segmenter = PsyduckSegmenter()
        print(segmenter.get_bounding_boxes(image))


if __name__ == '__main__':
    main()
