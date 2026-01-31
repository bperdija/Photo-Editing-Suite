import cv2
from .base import ImageTechnique

# This denoise technique was adapted from examples from 
# https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html

class Denoising(ImageTechnique):
    name = "Denoising"

    def __init__(self):
        # Default parameters
        self.h = 10
        self.h_color = 10
        self.template_window_size = 7
        self.search_window_size = 21

    def apply(self, image):
        return cv2.fastNlMeansDenoisingColored(
            image, None, 
            self.h, 
            self.h_color, 
            self.template_window_size, 
            self.search_window_size
        )

    def description(self):
        return (
            "Applies Non-local Means Denoising algorithm for colored images. "
            "Removes noise while preserving edges and fine details."
        )
    
    def configure_params(self):
        
        return True
