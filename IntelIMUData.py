from math import pi
import numpy as np

from IMUData import IMUData
from models import Pose, Reading, time_to_timestamp

NUM_BASELINE = 1000
class IntelIMUData(IMUData):
    def load_and_format(self):
        with open("./data/intel.txt") as readFile:
            content = readFile.read().splitlines()
            lines = [line.split() for line in content if line.startswith("ODOM")]
            # Get the first x, y, theta values and the time values.
            readings = np.array([list(map(float, line[1:4] + [line[7]])) for line in lines])
            x = readings[:, 0]
            y = readings[:, 1]
            theta = readings[:, 2]
            times = np.array([int(10*x[3])*10 for x in readings])

            return np.array([(x[i], y[i], theta[i]) for i in range(len(readings))]), times
    
    @staticmethod
    def progress_pose(prev_pose: Pose, reading: Reading) -> Pose:
        data = reading.get_data()
        return Pose(data[0], data[1], data[2])

    @staticmethod
    def get_cov_input_uncertainty(prev_pose: Pose, reading: Reading) -> np.ndarray:
        result = np.diag([1.0, 1.0, 1.0])
        result[0][2] = reading.get_data()[0] - prev_pose.x()
        result[1][2] = reading.get_data()[1] - prev_pose.y()
        return result

    @staticmethod
    def get_cov_change_matrix(prev_pose: Pose, reading: Reading) -> np.ndarray:
        return np.abs(np.diag([(0.01)**2, (0.01)**2, (0.2*pi/180)**2]))