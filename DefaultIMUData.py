from scipy.io import loadmat
import numpy as np
from IMUData import IMUData

NUM_BASELINE = 1000
class DefaultIMUData(IMUData):
    def load_and_format(self):
        imu_data = loadmat("./data/imu")
        encoder_data = loadmat("./data/speed")

        raw_omega_data = imu_data["IMU"]['DATAf'][0][0][5]
        baseline_omega = sum(raw_omega_data[0:NUM_BASELINE])/NUM_BASELINE

        raw_speed_data = encoder_data["Vel"]['speeds'][0][0][0]
        baseline_speed = sum(raw_speed_data[0:NUM_BASELINE])/NUM_BASELINE

        omega = np.array([x - baseline_omega for x in raw_omega_data])
        speed = np.array([x - baseline_speed for x in raw_speed_data])
        times = np.array([x for x in imu_data["IMU"]['times'][0][0][0]])
        return times, speed, omega
