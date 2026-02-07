import cv2
import numpy as np
from .base import ImageTechnique


class ColourMixing(ImageTechnique):
    name = "Colour Mixing"

    def __init__(self):
        # Default parameters (0 = no change, range -100 to +100)
        self.reds = 0
        self.greens = 0
        self.blues = 0
        self.yellows = 0
        self.oranges = 0
        self.aquas = 0
        self.purples = 0
        self.magentas = 0

    def apply(self, image):
        # If all adjustments are 0, return the original image unchanged
        if (
            self.reds == 0
            and self.greens == 0
            and self.blues == 0
            and self.yellows == 0
            and self.oranges == 0
            and self.aquas == 0
            and self.purples == 0
            and self.magentas == 0
        ):
            return image

        # Convert to HSV for targeted color adjustments
        # Why HSV?, because we can easily adjust:
        # H (Hue) = color type (red, green, blue, yellow)
        # S (Saturation) = how “pure” the color is
        # By changing S only where H is in a target range, we adjust that color only.
        # Taken from: https://stackoverflow.com/questions/8535650/how-to-change-saturation-values-with-opencv
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Hue ranges in OpenCV are 0-180
        # See colour slider: https://www.selecolor.com/en/hsv-color-picker/
        # This is a range which is later used for hue_min and hue_max
        color_ranges = {
            "reds": [(0, 10), (170, 180)],
            "oranges": [(10, 20)],
            "yellows": [(20, 35)],
            "greens": [(35, 95)],
            "aquas": [(80, 100)],
            "blues": [(95, 140)],
            "purples": [(130, 150)],
            "magentas": [(150, 170)],
        }

        adjustments = {
            "reds": self.reds,
            "oranges": self.oranges,
            "yellows": self.yellows,
            "greens": self.greens,
            "aquas": self.aquas,
            "blues": self.blues,
            "purples": self.purples,
            "magentas": self.magentas,
        }

        # Saturation channel index
        # Work only on saturation channel
        # hsv[:, :, 0] = Hue
        # hsv[:, :, 1] = Saturation
        # hsv[:, :, 2] = Value (brightness)
        # The colons are to take all rows, all columns (of channel 1)
        sat = hsv[:, :, 1]

        # For each color, build a mask & adjust only those pixels
        for color_name, adjustment in adjustments.items():
            # Skip any slider that’s still at 0.
            if adjustment == 0:
                continue

            # Build a hue mask for the color range(s)
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for hue_min, hue_max in color_ranges[color_name]:
                lower = np.array([hue_min, 10, 10], dtype=np.uint8)
                upper = np.array([hue_max, 255, 255], dtype=np.uint8)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv.astype(np.uint8), lower, upper))

            # Convert slider value to saturation multiplier
            # Example: 
            # Slider = +25 → multiplier = 1.25 (increase saturation)
            # Slider = -50 → multiplier = 0.5 (decrease saturation)
            multiplier = 1.0 + (adjustment / 100.0)

            # Apply saturation adjustment only where mask is active
            adjusted_sat = np.clip(sat * multiplier, 0, 255)
            sat = np.where(mask > 0, adjusted_sat, sat)

        # Put saturation back & convert back to BGR. This returns a normal BGR image that OpenCV our GUI expects
        hsv[:, :, 1] = sat
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def description(self):
        return (
            "Adjust the saturation of specific colors (Reds, Oranges, Yellows, Greens, Aquas, "
            "Blues, Purples, Magentas) independently. Positive values increase saturation, "
            "negative values decrease it."
        )

