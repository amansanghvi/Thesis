from typing import Any, List
from models import Pose, Position, timestamp_to_time
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from math import pi, sin, cos
from collections.abc import Sequence

POINTS_PER_SCAN = 361

class Lidar(Sequence):

    POINTS_PER_SCAN = POINTS_PER_SCAN
    _angles = [-pi/2 + i*pi/360 for i in range(0, POINTS_PER_SCAN)] # [-pi/2, pi/2]
    _scans = np.empty(0)
    _times = np.empty(0)

    _matlab: Any = None

    def __init__(self, filename: str, engine): # Matlab engine
        super().__init__()
        lidarData = loadmat(filename)
        self._scans = np.array(
            [0.01*(x & 0x1FFF) for x in lidarData['dataL']['Scans'][0][0]]
        ).transpose()
        self._times = np.array([
            x for x in lidarData['dataL']['times'][0][0][0]]
        )
        self._matlab = engine

    def __getitem__(self, idx: int):
        if not (isinstance(idx, int) or isinstance(idx, np.int64)):
            raise Exception("Invalid attribute: " + str(idx))
        return Scan(self._scans[idx], self._times[idx])
    
    def timestamp_for_idx(self, idx: int) -> int:
        if not (isinstance(idx, int) or isinstance(idx, np.int64)):
            raise Exception("Invalid attribute: " + str(idx))
        return self._times[idx]
    
    def get_at_time(self, timestamp: int):
        idx = self._times.searchsorted(timestamp)
        if idx == 0 or idx == len(self._times):
            return None
        return self[idx]

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
            ax.title.set_text("Frame: " + str(i) + "/" + str(len(self)) + " (" + ("%.3f" % 0.0001*self._times[i]) + ")")
            x = scan.x()
            y = scan.y()
            sc.set_offsets([[x[i], y[i]] for i in range(0, Lidar.POINTS_PER_SCAN)])
            fig.canvas.draw_idle()
            plt.pause(0.0001)
        plt.waitforbuttonpress()

class Scan:
    _x = np.array()
    _y = np.array()
    _time = 0.0
    _timestamp = 0
    
    def __init__(self, scan_data: np.ndarray, timestamp: int):
        if isinstance(scan_data, np.ndarray):
            self._x = np.array([scan_data[i]*cos(Lidar._angles[i]) for i in range(0, Lidar.POINTS_PER_SCAN)])
            self._y = np.array([scan_data[i]*sin(Lidar._angles[i]) + 0.46 for i in range(0, Lidar.POINTS_PER_SCAN)])
        self._time = timestamp_to_time(timestamp)
        self._timestamp = timestamp

    def x(self) -> np.ndarray:
        return self._x

    def y(self) -> np.ndarray:
        return self._y

    def time(self) -> float:
        return self._time

    def timestamp(self) -> int:
        return self._timestamp
    
    def __getitem__(self, idx: int) -> Position:
        if not isinstance(idx, int):
            raise Exception("Invalid attribute: " + str(idx))
        return Position(self._x[idx], self._y[idx])
    
    def __len__(self) -> int:
        return len(self._x)
    
    def __str__(self) -> str:
        return "Scan Class: " + str(len(self._x)) + " points at timestamp: " + str(self._time)
        
    def __iter__(self):
        self.n = 0
    def __next__(self):
        self.n += 1
        if self.n == len(self):
            raise StopIteration
        return self[self.n]
    
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
        scan = Scan(None, self._timestamp)
        scan._x = result[0]
        scan._y = result[1]
        
        return scan

    def show(self):
        plt.figure()
        plt.scatter(self._x, self._y, s=2)
        plt.xlim(0, 20)
        plt.ylim(-10, 10)
        plt.show(block=False)
