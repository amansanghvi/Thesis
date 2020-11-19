from IMUData import IMUData
import numpy as np
from collections.abc import Sequence
from models import Reading

class IMU(Sequence):
    def __init__(self, data: IMUData):
        self._data = data.get_data()
        self._times = data.get_times()
        self._progress_fnc = data.progress_pose
    
    def __getitem__(self, idx: int) -> Reading:
        if not (isinstance(idx, int) or isinstance(idx, np.int64)):
            raise Exception("Invalid attribute: " + str(idx) + " (" + str(type(idx)) + ")")
        return Reading(self._data[idx], self._times[idx], self._progress_fnc)
    
    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        return "IMU Class: " + str(len(self._data)) + " readings"
