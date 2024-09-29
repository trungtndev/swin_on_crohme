from typing import List

import cv2
import numpy as np
import torch
from torch import Tensor

from torchvision import transforms as tr

#data augmentation
rand_aug = tr.RandomChoice([
    tr.RandomRotation(degrees=2),
    tr.RandomRotation(degrees=10),
    tr.RandomRotation(degrees=8),
    tr.RandomRotation(degrees=15),
    tr.RandomAffine(degrees=5, translate=(0.3, 0.3), scale=(0.7, 1.3), shear=5),
    tr.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    tr.RandomAffine(degrees=15, translate=(0.4, 0.4), scale=(0.6, 1.4), shear=25),
    tr.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
])


class ScaleToLimitRange:
    def __init__(self, w_lo: int, w_hi: int, h_lo: int, h_hi: int) -> None:
        assert w_lo <= w_hi and h_lo <= h_hi
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.h_lo = h_lo
        self.h_hi = h_hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        r = h / w
        lo_r = self.h_lo / self.w_hi
        hi_r = self.h_hi / self.w_lo
        assert lo_r <= h / w <= hi_r, f"img ratio h:w {r} not in range [{lo_r}, {hi_r}]"

        scale_r = min(self.h_hi / h, self.w_hi / w)
        if scale_r < 1.0:
            # one of h or w highr that hi, so scale down
            img = cv2.resize(
                img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR
            )
            return img

        scale_r = max(self.h_lo / h, self.w_lo / w)
        if scale_r > 1.0:
            # one of h or w lower that lo, so scale up
            img = cv2.resize(
                img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR
            )
            return img

        # in the rectangle, do not scale
        assert self.h_lo <= h <= self.h_hi and self.w_lo <= w <= self.w_hi
        return img


class ScaleAugmentation:
    def __init__(self, lo: float, hi: float) -> None:
        assert lo <= hi
        self.lo = lo
        self.hi = hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        k = np.random.uniform(self.lo, self.hi)
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        return img
