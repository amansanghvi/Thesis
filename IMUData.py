from abc import abstractclassmethod
from typing import Tuple
import numpy as np


class IMUData:
    def __init__(self):
        super().__init__()
        self._times, self._speeds, self._yaws = self.load_and_format()

    @abstractclassmethod
    def load_and_format(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
    
    def get_times(self):
        return self._times

    def get_speeds(self):
        return self._speeds
    
    def get_yaws(self):
        return self._yaws

