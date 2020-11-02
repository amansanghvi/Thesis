import numpy as np
from imu import Reading
from models import Pose
from math import cos, sin
import matplotlib.pyplot as plt

MAP_LENGTH = 10 # metres
CELLS_IN_ROW = 100
CELL_SIZE = MAP_LENGTH / CELLS_IN_ROW

class Robot:
    _x = [0]
    _y = [0]
    _theta = [0]
    def __init__(self):
        pass
    
    def __getitem__(self, idx: int):
        if not isinstance(idx, int):
            raise Exception("Invalid attribute: " + str(idx))
        return self._map[idx]

    def x(self) -> np.ndarray:
        return self._x

    def y(self) -> np.ndarray:
        return self._y

    def theta(self) -> np.ndarray:
        return self._theta

    def get_latest_pose(self) -> Pose:
        return Pose(self._x[-1], self._y[-1], self._theta[-1])
    
    def imu_update(self, reading: Reading) -> Pose:
        prev_pose = self.get_latest_pose()
        
        new_theta = prev_pose.theta() + reading.dt()*reading.omega()
        new_x = prev_pose.x() + reading.dt()*cos(new_theta)*reading.speed()
        new_y = prev_pose.y() + reading.dt()*sin(new_theta)*reading.speed()
        
        self._x.append(new_x)
        self._y.append(new_y)
        self._theta.append(new_theta)

        return Pose(new_x, new_y, new_theta)

    def map_update(self):
        pass

    def __str__(self) -> str:   
        return "Robot at position: " + str(self.get_latest_pose())

    def show(self):
        plt.figure()
        plt.plot(self._x, self._y)
        plt.show(block=False)
