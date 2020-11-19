from IMUData import IMUData
from typing import Optional
import numpy as np
from collections.abc import Sequence
from models import Reading

NUM_BASELINE = 1000
class IMU(Sequence):
    _omega = np.empty(0) # Angular velocity (rad/s)
    _speed = np.empty(0) # velocities (m/s)
    _times = np.empty(0)
    def __init__(self, data: IMUData):
        self._omega = data.get_yaws()
        self._speed = data.get_speeds()
        self._times = data.get_times()
    
    def __getitem__(self, idx: int) -> Reading:
        if not (isinstance(idx, int) or isinstance(idx, np.int64)):
            raise Exception("Invalid attribute: " + str(idx) + " (" + str(type(idx)) + ")")
        return Reading(self._omega[idx], self._speed[idx], self._times[idx])
    
    def get_at_time(self, timestamp: int) -> Optional[Reading]:
        idx = self._times.searchsorted(timestamp, side="right")
        if idx == len(self._times):
            return None
        return self[idx]
    
    def __len__(self) -> int:
        return len(self._omega)

    def __str__(self) -> str:
        return "IMU Class: " + str(len(self._omega)) + " readings"
