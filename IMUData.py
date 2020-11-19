from abc import abstractclassmethod
from models import Pose, Reading
from typing import Tuple
import numpy as np


class IMUData:
    def __init__(self):
        super().__init__()
        self._data, self._times = self.load_and_format()
        assert(self._data.shape[0] > 2)
        assert(self._data.shape[1] == 2)
        assert(len(self._times.shape) == 1)

    @abstractclassmethod
    def load_and_format(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
    
    @staticmethod
    @abstractclassmethod
    def progress_pose(prev_pose: Pose, reading: Reading) -> Pose:
        pass

    def get_data(self):
        return self._data

    def get_times(self):
        return self._times

