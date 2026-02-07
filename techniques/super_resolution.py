import cv2
import numpy as np
from PIL import Image

from .base import ImageTechnique


class SuperResolution(ImageTechnique):
    name = "Super Resolution"

    MODEL_URL = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

    def __init__(self):
        super().__init__()
        self._model = None

    def description(self) -> str:
        return "Upscale using upsampling, OpenCV, or deep learning (ESRGAN) when available."

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply super resolution using the deep learning method by default."""
        try:
            return self.apply_deep_learning(image)
        except (ImportError, Exception) as e:
            print(f"Deep learning method failed: {e}. Falling back to OpenCV method.")
            return self.apply_opencv_method(image)

    def _crop_or_pad_to_multiple_of_4(self, image: np.ndarray, pad: bool = False) -> np.ndarray:
        height, width = image.shape[:2]
        target_height = height - (height % 4)
        target_width = width - (width % 4)
        if target_height == height and target_width == width:
            return image
        if pad:
            pad_h = (4 - (height % 4)) % 4
            pad_w = (4 - (width % 4)) % 4
            return cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        return image[:target_height, :target_width]

    # via upsampling: https://www.geeksforgeeks.org/python/spatial-resolution-down-sampling-and-up-sampling-in-image-processing/
    def apply_upsampling(self, image: np.ndarray, scale: int = 2) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        def upsample_channel(channel: np.ndarray) -> np.ndarray:
            m, n = channel.shape
            out = np.zeros((m * scale, n * scale), dtype=channel.dtype)

            for i in range(m):
                for j in range(n):
                    out[i * scale, j * scale] = channel[i, j]

            for i in range(1, m * scale, scale):
                out[i:i + (scale - 1), :] = out[i - 1, :]

            for i in range(0, m * scale):
                for j in range(1, n * scale, scale):
                    out[i, j:j + (scale - 1)] = out[i, j - 1]

            return out

        if image.ndim == 2:
            return upsample_channel(image)

        channels = [upsample_channel(image[:, :, idx]) for idx in range(image.shape[2])]
        return np.stack(channels, axis=2)

    def apply_opencv_method(self, image: np.ndarray, scale: int = 2) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        height, width = image.shape[:2]
        return cv2.resize(
            image,
            (width * scale, height * scale),
            interpolation=cv2.INTER_CUBIC
        )

    # Via: https://www.tensorflow.org/hub/tutorials/image_enhancing
    def apply_deep_learning(self, image: np.ndarray) -> np.ndarray:
        """
        Apply ESRGAN-based super resolution using TensorFlow Hub.
        Based on: https://www.tensorflow.org/hub/tutorials/image_enhancing
        """
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except Exception as exc:
            raise ImportError("TensorFlow and tensorflow_hub are required for the deep learning method.") from exc

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image: crop to multiple of 4
        hr_image = self._preprocess_image(rgb)
        
        # Load model (cache it for reuse)
        if self._model is None:
            print("Loading ESRGAN model from TensorFlow Hub...")
            self._model = hub.load(self.MODEL_URL)
        
        # Run super resolution
        fake_image = self._model(hr_image)
        fake_image = tf.squeeze(fake_image)
        
        # Clip and convert to uint8
        fake_image = tf.clip_by_value(fake_image, 0, 255)
        fake_image = tf.cast(fake_image, tf.uint8).numpy()
        
        # Convert back to BGR for OpenCV
        return cv2.cvtColor(fake_image, cv2.COLOR_RGB2BGR)
    
    def _preprocess_image(self, rgb_image: np.ndarray):
        """
        Preprocess image for ESRGAN model.
        Crops to multiple of 4 and converts to float32 tensor.
        """
        import tensorflow as tf
        
        # Crop to bounding box that's a multiple of 4
        height, width = rgb_image.shape[:2]
        hr_size_h = (height // 4) * 4
        hr_size_w = (width // 4) * 4
        hr_image = rgb_image[:hr_size_h, :hr_size_w]
        
        # Convert to float32 and add batch dimension
        hr_image = tf.cast(hr_image, tf.float32)
        return tf.expand_dims(hr_image, 0)
