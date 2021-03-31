from collections.abc import Sequence
from math import cos, sin
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from LidarData import LidarData
from models import Pose, Position, timestamp_to_time

POINTS_PER_SCAN = 361

class Lidar(Sequence):

    _scans = np.empty(0)
    _times = np.empty(0)

    _matlab: Any = None

    def __init__(self, data: LidarData, engine): # Matlab engine
        super().__init__()

        self._scans = data.get_scans()
        self._times = data.get_times()
        self._angles = data.get_angles()
        self._matlab = engine

    def __getitem__(self, idx: int):
        if not (isinstance(idx, int) or isinstance(idx, np.int64)):
            raise Exception("Invalid attribute: " + str(idx))
        return Scan(self._scans[idx], self._angles, self._times[idx])
    
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
    
    def angles(self, idx: int) -> float:
        return self._angles[idx]

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
            sc.set_offsets([[x[i], y[i]] for i in range(len(self._angles))])
            fig.canvas.draw_idle()
            plt.pause(0.0001)
        plt.waitforbuttonpress()

class Scan:
    _x = np.array([])
    _y = np.array([])
    _time = 0.0
    _timestamp = 0
    
    def __init__(self, ranges: np.ndarray, angles: np.ndarray, timestamp: int):
        if isinstance(ranges, np.ndarray):
            self._x = np.array([ranges[i]*cos(angles[i]) for i in range(len(ranges))])
            self._y = np.array([ranges[i]*sin(angles[i]) for i in range(len(ranges))])
        self._timestamp = timestamp

    def x(self) -> np.ndarray:
        return self._x

    def y(self) -> np.ndarray:
        return self._y

    def timestamp(self) -> int:
        return self._timestamp
    
    def __getitem__(self, idx: int) -> Position:
        if not isinstance(idx, int):
            raise Exception("Invalid attribute: " + str(idx))
        return Position(self._x[idx], self._y[idx])
    
    def __len__(self) -> int:
        return len(self._x)
    
    def __str__(self) -> str:
        return "Scan Class: " + str(len(self._x)) + " points at timestamp: " + str(self._timestamp)
        
    def __iter__(self) -> Position:
        self.n = 0
        return self[self.n]
    def __next__(self) -> Position:
        self.n += 1
        if self.n == len(self):
            raise StopIteration
        return self[self.n]
    
    def from_global_reference(self, frame: Pose): 
        # returns Scan with coordinates of points from global frame of reference 
        # given lidar pose in global reference frame.
        t_mat = [ # transformation matrix
            [cos(frame.theta()), -sin(frame.theta()), frame.x()], 
            [sin(frame.theta()),  cos(frame.theta()), frame.y()], 
            [                  0,                  0,         1],
        ]
        curr_positions = np.vstack(
            (self._x, self._y, [1 for _ in range(0, len(self._x))])
        )

        result = np.matmul(t_mat, curr_positions)
        scan = Scan(None, None, self._timestamp)
        scan._x = result[0]
        scan._y = result[1]
        
        return scan

    def show(self):
        plt.figure()
        plt.scatter(self._x, self._y, s=2)
        plt.xlim(0, 20)
        plt.ylim(-10, 10)
        plt.show(block=False)
