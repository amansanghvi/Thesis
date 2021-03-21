from collections.abc import Sequence

import numpy as np

from IMUData import IMUData
from models import Reading


class IMU(Sequence):
    def __init__(self, data: IMUData):
        self._data = data.get_data()
        self._times = data.get_times()
        self._progress_fnc = data.progress_pose
        self._get_cov_input_uncertainty = data.get_cov_input_uncertainty
        self._get_cov_change_matrix = data.get_cov_change_matrix
    
    def __getitem__(self, idx: int) -> Reading:
        if not (isinstance(idx, int) or isinstance(idx, np.int64)):
            raise Exception("Invalid attribute: " + str(idx) + " (" + str(type(idx)) + ")")
        return Reading(self._data[idx], self._times[idx], self._progress_fnc, self._get_cov_change_matrix, self._get_cov_input_uncertainty)
    
    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        return "IMU Class: " + str(len(self._data)) + " readings"
