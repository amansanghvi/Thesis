from abc import abstractclassmethod
from typing import Tuple
import numpy as np


class LidarData:
    def __init__(self):
        super().__init__()
        self._times, self._scans, self._angles = self.load_and_format()

    @abstractclassmethod
    def load_and_format(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
    
    def get_times(self):
        return self._times

    def get_scans(self):
        return self._scans
    
    def get_angles(self):
        return self._angles

