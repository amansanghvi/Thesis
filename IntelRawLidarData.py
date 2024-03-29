from math import pi
from typing import Tuple

import numpy as np

from LidarData import LidarData

POINTS_PER_SCAN = 180

class IntelRawLidarData(LidarData):

    def load_and_format(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with open("./data/intel_raw.log") as readFile:
            content = readFile.read().splitlines()
            lines = [line.split() for line in content if line.startswith("FLASER")]
            scans = np.array([list(map(float, line[2:(POINTS_PER_SCAN + 2)])) for line in lines])
            times = np.array([int(1000*float(line[-1]))*10 for line in lines])
            sorted_times, idxs = np.unique(times, return_index=True)
            # [-pi/2, pi/2]
            angles = np.array([-pi/2 + i*pi/179 for i in range(POINTS_PER_SCAN)]) 
            return sorted_times, scans[idxs], angles
        
