from scipy.io import loadmat
import numpy as np
from collections.abc import Sequence

NUM_BASELINE = 1000
class IMU(Sequence):
    _omega = [] # Angular velocity (rad/s)
    _speed = [] # velocities (m/s)
    _times = []
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
        self._times = np.array([0.0001*x for x in imu_data["IMU"]['times'][0][0][0]])
    
    def __getitem__(self, idx: int):
        if not isinstance(idx, int):
            raise Exception("Invalid attribute: " + str(idx))
        dt = self._times[idx] - self._times[idx-1] if idx > 0 else 0.005 # 5ms
        return Reading(self._omega[idx], self._speed[idx], dt)

    def __len__(self) -> int:
        return len(self._omega)

    def __str__(self) -> str:
        return "IMU Class: " + str(len(self._omega)) + " readings"

class Reading:
    _omega = 0
    _speed = 0
    _dt = 0

    def __init__(self, omega: float, speed: float, dt: float):
        self._dt = dt
        self._omega = omega
        self._speed = speed
    
    def omega(self):
        return self._omega

    def speed(self):
        return self._speed

    def dt(self):
        return self._dt


