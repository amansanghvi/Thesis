from math import pi
from typing import Tuple

import numpy as np

from LidarData import LidarData

POINTS_PER_SCAN = 180

class IntelLidarData(LidarData):

    def load_and_format(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with open("./data/intel.txt") as readFile:
            content = readFile.read().splitlines()
            lines = [line.split() for line in content if line.startswith("FLASER")]
            scans = np.array([list(map(float, line[2:(POINTS_PER_SCAN + 2)])) for line in lines])
            times = np.array([int(10*float(line[-3]))*10 for line in lines])
            # [-pi/2, pi/2]
            angles = np.array([-pi/2 + i*pi/179 for i in range(POINTS_PER_SCAN)]) 
            return times, scans, angles
        
