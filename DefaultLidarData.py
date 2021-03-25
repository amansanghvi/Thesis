from scipy.io import loadmat
import numpy as np
from math import pi
from LidarData import LidarData

POINTS_PER_SCAN = 361

class DefaultLidarData(LidarData):

    def load_and_format(self):
        lidarData = loadmat("./data/lidar")
        scans = np.array(
            [0.01*(x & 0x1FFF) for x in lidarData['dataL']['Scans'][0][0]]
        ).transpose()
        times = np.array([
            x for x in lidarData['dataL']['times'][0][0][0]]
        )
        angles = np.array([-pi/2 + i*pi/360 for i in range(POINTS_PER_SCAN)]) # [-pi/2, pi/2]
        return times, scans, angles
        
