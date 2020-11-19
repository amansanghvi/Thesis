from models import time_to_timestamp
import numpy as np
from math import pi
from LidarData import LidarData

POINTS_PER_SCAN = 180

class DefaultLidarData(LidarData):

    def load_and_format(self):
        with open("./data/intel_LASER_.txt") as readFile:
            content = readFile.read().splitlines()
            scans = np.array([list(map(float, line.split())) for line in content])
            times = np.array([time_to_timestamp(x) for x in range(len(scans))])
            # [-pi/2, pi/2]
            angles = np.array(-pi/2 + i*pi/179 for i in range(POINTS_PER_SCAN)) 
            return times, scans, angles
        
