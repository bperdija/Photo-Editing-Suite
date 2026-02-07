import cv2
import numpy as np

from .base import ImageTechnique


class WhiteBalancing(ImageTechnique):
    name = "White Balancing"

    def description(self) -> str:
        return "White balance via Gray World Assumption."

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Default white balance (Gray World Assumption)."""
        return self.apply_gray_world(image)

    # This algorithm was adapted from: 
    # https://pippin.gimp.org/image-processing/chapter-automaticadjustments.html
    def apply_gray_world(self, image: np.ndarray) -> np.ndarray:
        """Apply gray world white balance in CIE Lab space."""
        if image is None or image.size == 0:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Calculate Color Cast to measure how far the average colore deviates from neutral
        # Positive a_shift: Image is too red
        # Negative a_shift: Image is too green
        # Same logic for blue-yellow on b channel
        # OpenCV Lab ranges: L [0,255], a/b [0,255] with 128 as neutral
        l_scaled = l_channel / 255.0 * 100.0
        a_shift = np.mean(a_channel - 128.0)
        b_shift = np.mean(b_channel - 128.0)

        # Apply shift scaled by luminance
        scale = (l_scaled / 100.0) * 1.1
        a_channel = (a_channel - 128.0) + (-a_shift * scale) + 128.0
        b_channel = (b_channel - 128.0) + (-b_shift * scale) + 128.0

        a_channel = np.clip(a_channel, 0, 255)
        b_channel = np.clip(b_channel, 0, 255)

        balanced_lab = cv2.merge([
            np.clip(l_channel, 0, 255),
            a_channel,
            b_channel
        ]).astype(np.uint8)

        return cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2BGR)

