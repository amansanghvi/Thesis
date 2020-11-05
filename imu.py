from typing import Optional, Union
from scipy.io import loadmat
import numpy as np
from collections.abc import Sequence
from models import Reading, timestamp_to_time

NUM_BASELINE = 1000
class IMU(Sequence):
    _omega = np.empty(0) # Angular velocity (rad/s)
    _speed = np.empty(0) # velocities (m/s)
    _times = np.empty(0)
    def __init__(self, imu_file: str, encoder_file: str):
        super().__init__()
        imu_data = loadmat(imu_file)
        encoder_data = loadmat(encoder_file)

        raw_omega_data = imu_data["IMU"]['DATAf'][0][0][5]
        baseline_omega = sum(raw_omega_data[0:NUM_BASELINE])/NUM_BASELINE

        raw_speed_data = encoder_data["Vel"]['speeds'][0][0][0]
        baseline_speed = sum(raw_speed_data[0:NUM_BASELINE])/NUM_BASELINE

        self._omega = np.array([x - baseline_omega for x in raw_omega_data])
        self._speed = np.array([x - baseline_speed for x in raw_speed_data])
        self._times = np.array([x for x in imu_data["IMU"]['times'][0][0][0]])
    
    def __getitem__(self, idx: int) -> Optional[Reading]:
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
