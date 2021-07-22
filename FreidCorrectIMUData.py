from math import pi
import numpy as np

from IMUData import IMUData
from models import Pose, Reading, time_to_timestamp

NUM_BASELINE = 1000
class FreidCorrectIMUData(IMUData):
    def load_and_format(self):
        with open("./data/fr_correct.log") as readFile:
            content = readFile.read().splitlines()
            lines = [line.split() for line in content if line.startswith("ODOM")]
            # Get the first x, y, theta values and the time values.
            readings = np.array([list(map(float, line[1:4] + [line[9]])) for line in lines])
            x = -readings[:, 0]
            y = readings[:, 1]
            theta = readings[:, 2]
            times = np.array([int(1000*t[3])*10 for t in readings])
            sorted_times, idxs = np.unique(times, return_index=True)

            imu_data = np.array([(x[i], y[i], theta[i]) for i in range(len(readings))])
            calibrated_mean = np.mean(imu_data[0:5], axis=0)
            imu_pos = np.array([reading - calibrated_mean for reading in imu_data])
            real_times = np.column_stack((sorted_times, sorted_times, sorted_times))
            imu_vel = 1e4*np.diff(imu_pos[idxs], axis=0)/np.diff(real_times, axis=0)
            adj_vel = np.array([[v[0], v[1], 0.0] if abs(v[2]) < 0.08 else v for v in imu_vel])
            # print("Velocities")
            # for vel in imu_vel:
            #     print(vel) 

            return np.vstack(([0.0, 0.0, 0.0], imu_vel)), sorted_times

    @staticmethod
    def progress_pose(prev_pose: Pose, reading: Reading) -> Pose:
        data = reading.get_data()
        dt = reading.dt()/1e4
        return Pose(
            prev_pose.x() + data[0]*dt, 
            prev_pose.y() + data[1]*dt, 
            prev_pose.theta() + data[2]*dt
        )

    @staticmethod
    def get_cov_change_matrix(prev_pose: Pose, reading: Reading) -> np.ndarray:
        result = np.diag([1.0, 1.0, 1.0])
        return result

    @staticmethod
    def get_cov_input_uncertainty(prev_pose: Pose, reading: Reading) -> np.ndarray:
        dt = reading.dt()/1e4
        return np.abs(np.diag([
            (0.02 + 0.01*abs(reading._data[0])*dt)**2,
            (0.02 + 0.01*abs(reading._data[1])*dt)**2,
            (0.2*pi/180 + 0.02*abs(reading._data[2])*dt)**2
        ])) 
