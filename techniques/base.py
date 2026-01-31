# This is a template I will use for all techniques. 
from abc import ABC, abstractmethod
import numpy as np

class ImageTechnique(ABC):
    name: str

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def description(self) -> str:
        pass
    
    def configure_params(self):
        """Override this method to allow user parameter configuration.
        Returns True if configuration is available, False otherwise."""
        return False