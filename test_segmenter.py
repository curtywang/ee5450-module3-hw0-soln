from typing import List, Dict, Tuple
import pytest
import numpy as np
import cv2
from segmenter import PsyduckSegmenter, PikachuSegmenter


PIKACHU_BOUNDS = {  # XYWH format
    'pokemon-cave.jpg': (468, 336, 258, 421),
    'pokemon-forest.jpg': (602, 9, 186, 492),
    'pokemon-pika-ketchup.jpg': (348, 0, 675, 768),
}

PSYDUCK_BOUNDS = [
    (223, 220, 59, 89),
    (306, 201, 64, 85),
    (352, 80, 57, 85),
    (265, 44, 67, 103),
]


def overlap(xywh1: Tuple[int, int, int, int], xywh2: Tuple[int, int, int, int]) -> int:
    """
    Compute overlap of two bounds
    :param xywh1:
    :param xywh2:
    :return:
    """
    rect1 = np.zeros((1200, 1200))
    rect2 = np.zeros((1200, 1200))
    x1, y1, w1, h1 = xywh1
    x2, y2, w2, h2 = xywh2
    rect1[x1:x1 + w1, y1:y1 + h1] = 1
    rect2[x2:x2 + w2, y2:y2 + h2] = 1
    return np.sum((rect1 + rect2) > 1)


def test_pikachu():
    for file_name, file_bounds in PIKACHU_BOUNDS.items():
        the_segmenter = PikachuSegmenter()
        test_image: np.ndarray = cv2.imread(f'sample/{file_name}')
        out_list: List[Tuple[int, int, int, int]] = the_segmenter.get_bounding_boxes(test_image)
        for detected_bounds in out_list:
            min_overlap = detected_bounds[2] * detected_bounds[3] * 0.5
            assert overlap(file_bounds, detected_bounds) > min_overlap


def test_psyduck():
    the_segmenter = PsyduckSegmenter()
    test_image = cv2.imread('sample/pokemon-psyducks.jpg')
    out_list = the_segmenter.get_bounding_boxes(test_image)
    for detected_bounds in out_list:
        min_overlap = detected_bounds[2] * detected_bounds[3] * 0.5
        assert max([overlap(file_bounds, detected_bounds) for file_bounds in PSYDUCK_BOUNDS]) > min_overlap
