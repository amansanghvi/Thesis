from typing import List
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

class Position:
    x = 0
    y = 0
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

class Scan:
    _x = []
    _y = []
    _time = 0.0
    
    def __init__(self, scan_data: np.ndarray, timestamp: int):
        if isinstance(scan_data, np.ndarray):
            self._x = np.array([scan_data[i]*cos(Lidar._angles[i]) for i in range(0, Lidar.POINTS_PER_SCAN)])
            self._y = np.array([scan_data[i]*sin(Lidar._angles[i]) + 0.46 for i in range(0, Lidar.POINTS_PER_SCAN)])
        self._time = timestamp
        
    def x(self) -> np.ndarray:
        return self._x

    def y(self) -> np.ndarray:
        return self._y

    def time(self) -> float:
        return self._time
    
    def __getitem__(self, idx: int) -> Position:
        if not isinstance(idx, int):
            raise Exception("Invalid attribute: " + str(idx))
        return Position(self._x[idx], self._y[idx])
    
    def __len__(self) -> int:
        return len(self._x)
    
    def __str__(self) -> str:
        return "Scan Class: " + str(len(self._x)) + " points at timestamp: " + str(self._time)
    
    def from_global_reference(self, frame: Pose): 
        # returns coordinates of points from global frame of reference 
        # given lidar pose in global reference frame.
        t_mat = [ # transformation matrix
            [cos(frame.theta()), -sin(frame.theta()), frame.x()], 
            [ sin(frame.theta()), cos(frame.theta()), frame.y()], 
            [                  0,                  0,         1],
        ]
        curr_positions = np.vstack(
            (self._x, self._y, [1 for _ in range(0, len(self._x))])
        )

        result = np.matmul(t_mat, curr_positions)
        scan = Scan(None, self._time)
        scan._x = result[0]
        scan._y = result[1]
        
        return scan

    def show(self):
        plt.figure()
        plt.scatter(self._x, self._y, s=2)
        plt.xlim(0, 20)
        plt.ylim(-10, 10)
        plt.show(block=False)
