from lidar import Scan
import numpy as np
from imu import Reading
from gridmap import GridMap
from models import Pose
from math import cos, sin
import scipy
import matplotlib.pyplot as plt

MAP_LENGTH = 10 # metres
CELLS_IN_ROW = 100
CELL_SIZE = MAP_LENGTH / CELLS_IN_ROW

class Robot:
    _x = [0.0]
    _y = [0.0]
    _theta = [0.0]
    _map: GridMap
    _weight = [0.0]
    def __init__(self, num_particles, matlab):
        self._map = GridMap(matlab, 20)
        self._weight = [1.0/num_particles]

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

    def map_update(self, scan: Scan):
        scan_pose = self._map.get_scan_match(scan, self.get_latest_pose())
        p_z_given_pose_and_map = 1
        for beam in scan:
            if beam.x()**2 + beam.y()**2 < 7: # If not out of range
                dist_sq = 0.1 # Distance^2 from closest obstacle in grid map
                pr_z = self._map.get_pr_at() # TODO: This
                pass 
        K_sample_points = 10
        for i in range(0, K_sample_points):
            pass

    def __str__(self) -> str:   
        return "Robot at position: " + str(self.get_latest_pose())

    def show(self):
        plt.figure()
        plt.plot(self._x, self._y)
        plt.show(block=False)
