from models import Pose, Reading, time_to_timestamp
import numpy as np
from math import pi
from IMUData import IMUData

NUM_BASELINE = 1000
class DefaultIMUData(IMUData):
    def load_and_format(self):
        with open("./data/intel_ODO.txt") as readFile:
            content = readFile.read().splitlines()
            readings = np.array([list(map(float, line.split())) for line in content])
            dtheta = readings[:, 2]
            dx = readings[:, 0]
            dy = readings[:, 1]
            times = np.array([time_to_timestamp(x) for x in range(len(readings))])

            return [(dx[i], dy[i], dtheta[i], times[i]) for i in range(len(readings))]
    
    @staticmethod
    def progress_pose(prev_pose: Pose, reading: Reading):
        pass