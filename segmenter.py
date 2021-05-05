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
        # pokemon-cave.jpg BGRK 1: [ 55, 181, 239, 184]
        # pokemon-cave.jpg BGRK 2:
        # pokemon-forest.jpg BGRK 1: [ 80, 215, 243, 208]
        # pokemon-forest.jpg BGRK 2: [ 82, 212, 242, 206]
        # ketchup.jpg BGRK 1: [ 50, 202, 219, 190]
        # ketchup.jpg BGRK 2: [ 48, 204, 223, 192]
        self.THRESHOLD_LOW = (35, 114, 218, 118)
        self.THRESHOLD_HIGH = (82, 215, 243, 208)

    def enhance_image(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Enhance the image so that you can more easily threshold it, and create a grayscale image too.

        :param bgr_image: input image in BGR format
        :return: enhanced image in BGRK (K for grayscale) format
        """
        k_image: np.ndarray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        return np.concatenate((bgr_image, np.expand_dims(k_image, axis=2)), axis=2)

    def threshold_enhanced_image(self, enhanced_bgrk_image: np.ndarray) -> np.ndarray:
        """
        Threshold the enhanced image using the self.THRESHOLD BGRK constant

        :param enhanced_bgrk_image: input image in BGRK format
        :return: binary image for each channel BGRK
        """
        masks = []
        for channel, ch_thresh_low, ch_thresh_high in zip(list(range(len(self.THRESHOLD_LOW))),
                                                          self.THRESHOLD_LOW, self.THRESHOLD_HIGH):
            masks.append(cv2.inRange(enhanced_bgrk_image[:, :, channel], ch_thresh_low, ch_thresh_high))
        masks = np.stack(masks, axis=2)
        # for channel in range(masks.shape[2]):
        #     plt.imshow(masks[:, :, channel])
        #     plt.show()
        final_mask = np.bitwise_and.reduce(masks, axis=2)
        plt.imshow(final_mask)
        plt.show()
        return final_mask

    def clean_thresholded_image(self, thresholded_bgrk_image: np.ndarray) -> np.ndarray:
        """
        Uses morphological operations to clean up thresholded images

        :param thresholded_bgrk_image: input binary mask from thresholding
        :return: cleaned binary mask from thresholding
        """
        kernel = np.ones((9, 9), np.uint8)
        opening = cv2.morphologyEx(thresholded_bgrk_image, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        dilation = cv2.dilate(closing, kernel, iterations=5)
        return dilation

    def get_bounding_boxes(self, input_bgr_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Get bounding boxes (x_left, y_top, width, height) of each object found.

        :param input_bgr_image: input image in BGR format
        :return: list of (x_left, y_top, width, height) bounding boxes
        """
        enhanced_image = self.enhance_image(input_bgr_image)
        threshold_mask = self.threshold_enhanced_image(enhanced_image)
        cleaned_mask = self.clean_thresholded_image(threshold_mask)
        plt.imshow(cleaned_mask)
        plt.show()
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_LIST, 2)
        return [cv2.boundingRect(contour) for contour in contours]


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
