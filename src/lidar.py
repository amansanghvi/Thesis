from robot import Pose
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from math import pi, sin, cos
from collections.abc import Sequence

POINTS_PER_SCAN = 361

class Lidar(Sequence):

    POINTS_PER_SCAN = POINTS_PER_SCAN
    _angles = [-pi/2 + i*pi/360 for i in range(0, POINTS_PER_SCAN)] # [-pi/2, pi/2]
    _scans = []
    _times = []

    def __init__(self, filename: str):
        super().__init__()
        lidarData = loadmat(filename)
        self._scans = np.array(
            [0.01*(x & 0x1FFF) for x in lidarData['dataL']['Scans'][0][0]]
        ).transpose()
        self._times = np.array([
            0.0001*x for x in lidarData['dataL']['times'][0][0][0]]
        )

    def __getitem__(self, name: int):
        if not isinstance(name, int):
            raise Exception("Invalid attribute: " + str(name))
        return Scan(self._scans[name], self._times[name])

    def __len__(self) -> int:
        return len(self._scans)

    def __str__(self) -> str:
        return "Lidar Class: " + str(len(self._scans)) + " scans"

    def show_all(self):
        plt.ion()
        fig, ax = plt.subplots()
        sc = ax.scatter([], [], s = 1)
        plt.xlim(0, 20)
        plt.ylim(-10, 10)
        plt.draw()
        
        for i, scan in enumerate(self):
            ax.title.set_text("Frame: " + str(i) + "/" + str(len(self)) + " (" + ("%.3f" % self._times[i]) + ")")
            x = scan.x()
            y = scan.y()
            sc.set_offsets([[x[i], y[i]] for i in range(0, Lidar.POINTS_PER_SCAN)])
            fig.canvas.draw_idle()
            plt.pause(0.0001)
        plt.waitforbuttonpress()


class Scan:
    _x = []
    _y = []
    _time = 0.0
    def __init__(self, scan_data: list, timestamp: int):
        self._x = np.array([scan_data[i]*cos(Lidar._angles[i]) for i in range(0, Lidar.POINTS_PER_SCAN)])
        self._y = np.array([scan_data[i]*sin(Lidar._angles[i]) + 0.46 for i in range(0, Lidar.POINTS_PER_SCAN)])
        self._time = timestamp
        
    def x(self) -> list:
        return self._x

    def y(self) -> list:
        return self._y

    def time(self) -> float:
        return self._time
    
    def from_reference(frame: Pose): # returns coordinates of points from given frame of reference 
        # TODO: Multiply by transformation matrix
        pass

    def show(self):
        if Scan._fig == None:
            plt.ion()
            Scan._fig, Scan._axes = plt.subplots()
            Scan._plt_data = Scan._axes.scatter([], [], s=1)
            Scan._axes.set_xlim(0, 20)
            Scan._axes.set_ylim(-10, 10)
            plt.draw()
        plt.scatter(self._x, self._y, s=1)
        plt.xlim(0, 20)
        plt.ylim(-10, 10)
