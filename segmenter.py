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
        :return: enhanced image in HSVK (K for grayscale) format
        """
        pass

    def threshold_enhanced_image(self, enhanced_hsvk_image: np.ndarray) -> np.ndarray:
        """
        Threshold the enhanced image using the self.THRESHOLD HSVK constant

        :param enhanced_hsvk_image: input image in HSVK format
        :return: binary image for each channel HSVK
        """
        pass

    def clean_thresholded_image(self, thresholded_hsvk_image: np.ndarray) -> np.ndarray:
        """
        Uses morphological operations to clean up thresholded images

        :param thresholded_hsvk_image: input binary image for each channel (HSVK)
        :return: cleaned binary image for each channel (HSVK)
        """
        pass

    def get_combined_thresholded_image(self, cleaned_hsvk_image: np.ndarray) -> np.ndarray:
        """
        Combines the binary image from each channel (HSVK) to form a single binary mask.

        :param cleaned_hsvk_image: cleaned binary image for each channel (HSVK)
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
        # ketchup HSV 1: [ 27, 195, 221, 192]
        # ketchup HSV 2: [ 20, 202, 155, 117]
        # cave HSV 1:    [ 21, 196, 239, 184]
        # cave HSV 2:    [ 20, 197, 154, 118]
        # forest HSV 1:  [ 25, 172, 242, 207]
        # forest HSV 2:  [ 25, 169, 243, 208]
        self.THRESHOLD_LOW = (20, 169, 154, 117)
        self.THRESHOLD_HIGH = (26, 202, 243, 208)

    def enhance_image(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Enhance the image so that you can more easily threshold it, and create a grayscale image too.

        :param bgr_image: input image in BGR format
        :return: enhanced image in HSVK (K for grayscale) format
        """
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)  # shape is (H, W, ch)
        k_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)  # shape is (H, W)
        return np.concatenate((hsv_image, np.expand_dims(k_image, axis=2)), axis=2)

    def threshold_enhanced_image(self, enhanced_hsvk_image: np.ndarray) -> np.ndarray:
        """
        Threshold the enhanced image using the self.THRESHOLD HSVK constant

        :param enhanced_hsvk_image: input image in HSVK format
        :return: binary image for each channel HSVK
        """
        masks = []
        for channel, ch_thresh_low, ch_thresh_high in zip(range(len(self.THRESHOLD_LOW)),
                                                          self.THRESHOLD_LOW, self.THRESHOLD_HIGH):
            masks.append(cv2.inRange(enhanced_hsvk_image[:, :, channel], ch_thresh_low, ch_thresh_high))
        return np.stack(masks, axis=2)

    def clean_thresholded_image(self, thresholded_hsvk_image: np.ndarray) -> np.ndarray:
        """
        Uses morphological operations to clean up thresholded images

        :param thresholded_hsvk_image: input binary image for each channel (HSVK)
        :return: cleaned binary image for each channel (HSVK)
        """
        kernel_small = np.ones((5, 5), dtype=np.uint8)
        kernel_large = np.ones((11, 11), dtype=np.uint8)
        closing = cv2.morphologyEx(thresholded_hsvk_image, cv2.MORPH_CLOSE, kernel_large)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_small)
        return opening

    def get_combined_thresholded_image(self, thresholded_hsvk_image: np.ndarray) -> np.ndarray:
        """
        Combines the binary image from each channel (HSVK) to form a single binary mask.

        :param thresholded_hsvk_image: cleaned binary image for each channel (HSVK)
        :return: single binary mask to pass to cv2.findContours() and cv2.boundingRect()
        """
        return np.bitwise_and.reduce(thresholded_hsvk_image, axis=2)

    def get_bounding_boxes(self, input_bgr_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Get bounding boxes (x_left, y_top, width, height) of each object found.

        :param input_bgr_image: input image in BGR format
        :return: list of (x_left, y_top, width, height) bounding boxes
        """
        enhanced_image = self.enhance_image(input_bgr_image)
        thresholded_image = self.threshold_enhanced_image(enhanced_image)
        combined_mask = self.get_combined_thresholded_image(thresholded_image)
        cleaned_mask = self.clean_thresholded_image(combined_mask)

        contours, _ = cv2.findContours(cleaned_mask, 1, 2)

        contour_areas = np.array([cv2.contourArea(contour) for contour in contours])
        max_contour = np.argmax(contour_areas)

        return [cv2.boundingRect(contours[max_contour])]


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
