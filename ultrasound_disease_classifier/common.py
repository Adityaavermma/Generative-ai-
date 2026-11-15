"""Constants and image helpers shared by data prep, training and prediction."""

import cv2
import numpy as np

CLASSES = ["benign", "malignant", "normal"]

# Enhanced images are cached at this square size; training/prediction resize
# from here, so prediction must go through to_cache_size() too.
CACHE_SIZE = 384


def to_cache_size(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA)
