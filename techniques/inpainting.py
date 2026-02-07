import cv2
import numpy as np

from .base import ImageTechnique


# We will utilize the inpaint method from OpenCV
# Source: https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html
class Inpainting(ImageTechnique):
    name = "Inpainting"

    def description(self) -> str:
        return "Inpaint using a user-provided mask."

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image

    def apply_with_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        if mask is None or mask.size == 0:
            return image

        mask_uint8 = mask.astype(np.uint8)
        if mask_uint8.ndim == 3:
            mask_uint8 = cv2.cvtColor(mask_uint8, cv2.COLOR_BGR2GRAY)

        return cv2.inpaint(image, mask_uint8, 3, cv2.INPAINT_TELEA)
