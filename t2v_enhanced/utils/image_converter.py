import cv2
import numpy as np
from albumentations.augmentations.geometric import functional as F
from albumentations.core.transforms_interface import DualTransform

__all__ = ["ProportionalMinScale"]


class ProportionalMinScale(DualTransform):

    def __init__(
            self,
            width: int,
            height: int,
            interpolation: int = cv2.INTER_LINEAR,
            always_apply: bool = False,
            p: float = 1,
    ):
        super(ProportionalMinScale, self).__init__(always_apply, p)
        self.width = width
        self.height = height

    def apply(
            self, img: np.ndarray, width: int = 256, height: int = 256, interpolation: int = cv2.INTER_LINEAR, **params):
        h_img, w_img, _ = img.shape

        min_side = np.min([h_img, w_img])

        if (height/h_img)*w_img >= width:
            if h_img == min_side:
                return F.smallest_max_size(img, max_size=height, interpolation=interpolation)
            else:
                return F.longest_max_size(img, max_size=height, interpolation=interpolation)
        if (width/w_img)*h_img >= height:
            if w_img == min_side:
                return F.smallest_max_size(img, max_size=width, interpolation=interpolation)
            else:
                return F.longest_max_size(img, max_size=width, interpolation=interpolation)
        return F.longest_max_size(img, max_size=width, interpolation=interpolation)

    def get_params(self):
        return {"width": self.width, "height": self.height}

    def get_transform_init_args_names(self):
        return ("width", "height", "intepolation")
