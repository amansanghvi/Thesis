from lidar import Scan
import numpy as np
from imu import Reading
from gridmap import GridMap
from models import Pose
from math import cos, sin
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
        scan_pose, cov = self._map.get_scan_match(scan, self.get_latest_pose())
        cov[0][0] = 0.06
        cov[1][1] = 0.06
        cov[2][2] = 0.01

        K_sample_points = 20
        x, y, th = np.random.multivariate_normal([0, 0, 0], cov, K_sample_points)
        selected_points = [
            np.add(scan_pose, (x[i], y[i], th[i])) for i in range(len(x))
                if abs(x[i]) < 4*cov[0][0] and abs(y[i]) < 4*cov[1][1] and abs(th[i]) < 4*cov[2][2]
        ]
        
        means = [np.array((0, 0, 0)) for _ in range(len(selected_points))]
        norms = [0.0]*len(selected_points)
        for i in range(len(selected_points)):
            x_k = selected_points[i]
            pr_x_k = 0.6
            pr_z = 1.0
            for beam in scan.from_global_reference(x_k):
                if beam.x()**2 + beam.y()**2 < 7: # If not out of range
                    # TODO: Decide matching points.
                    pr_z *= self._map.get_pr_at(beam) 
            means[i] = np.add(means[i], x_k * pr_z * pr_x_k)
            norms[i] = norms[i] + pr_z * pr_x_k

    def __str__(self) -> str:   
        return "Robot at position: " + str(self.get_latest_pose())

    def show(self):
        plt.figure()
        plt.plot(self._x, self._y)
        plt.show(block=False)
