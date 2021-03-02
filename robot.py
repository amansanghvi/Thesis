from typing import List
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scipy

from gridmap import GridMap
from imu import Reading
from lidar import Scan
from models import Pose

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
        
        next_pose = reading.get_moved_pose(prev_pose)
        
        self._x.append(next_pose.x())
        self._y.append(next_pose.y())
        self._theta.append(next_pose.theta())

        return next_pose

    def map_update(self, scan: Scan):
        latest_pose = self.get_latest_pose()
        scan_pose, cov = self._map.get_scan_match(scan, latest_pose)
        print(scan_pose, cov)

        K_sample_points = 10
        guesses = np.random.multivariate_normal([0, 0, 0], cov, K_sample_points)
        x, y, th = guesses[:, 0], guesses[:, 1], guesses[:, 2]

        selected_points = [
            np.add(scan_pose, (x[i], y[i], th[i])) for i in range(len(x))
                if abs(x[i]) < 4*cov[0][0] and abs(y[i]) < 4*cov[1][1] and abs(th[i]) < 4*cov[2][2]
        ]
        print("# k-points: " + str(len(selected_points)))
        # TODO: Get the predicted position below
        predicted_odd_mean = np.add([latest_pose.x(), latest_pose.y(), latest_pose.theta()], [0.0, 0.0, 0.0])
        predicted_odd_cov = np.diag([0.02, 0.02, 0.03])
        
        ksample_weights = self._generate_sample_weight(selected_points, predicted_odd_mean, predicted_odd_cov, scan)
        
        mean = predicted_odd_mean
        norm = np.longdouble(0.0)
        for i in range(len(selected_points)):
            mean = np.add(mean, selected_points[i] * ksample_weights[i])
            norm = norm + ksample_weights[i]
        if (norm < 0.0000001 and norm > -0.0000001):
            if len(selected_points) == 0:
                mean = predicted_odd_mean
                sigma = predicted_odd_cov
            else:
                return
        else:
            mean = mean/norm
            sigma = np.zeros((3, 3), dtype=np.longdouble)
            for i in range(len(selected_points)):
                delta = np.add(selected_points[i], -mean)
                sigma = sigma + delta*delta.T*ksample_weights[i]
            sigma = sigma/norm

        print("norm: " + str(norm) + " mean: " + str(mean))
        # TODO: Make x a distribution and use sigma as the uncertainty
        mean_pose = Pose(mean[0], mean[1], mean[2])
        self._x.append(mean_pose.x())
        self._y.append(mean_pose.y())
        self._theta.append(mean_pose.theta())
        self._map.update(mean_pose, scan)

    
    def _generate_sample_weight(self, selected_points: np.ndarray, predicted_odd_mean, predicted_odd_cov, scan: Scan) -> List[float]:
        motion_model = scipy.multivariate_normal(predicted_odd_mean, predicted_odd_cov)
        ksample_weights = np.zeros(len(selected_points), dtype=np.longdouble)
        for i in range(len(selected_points)):
            x_k = selected_points[i]
            position_weight = motion_model.pdf(x_k)
            observation_weight = np.longdouble(1.0)
            adjusted_scan = scan.from_global_reference(Pose(x_k[0], x_k[1], x_k[2]))
            for j in range(len(adjusted_scan)):
                beam = adjusted_scan[j]
                if beam.x**2 + beam.y**2 < 7 and beam.x**2 + beam.y**2 > 0.01: # If not out of range
                    observation_weight += self._map.get_pr_at(beam)*10 # Arbitrary scaling applied (*10)
                else:
                    observation_weight += 5 # 0.5 * 10
            ksample_weights[i] = observation_weight * position_weight
        return ksample_weights

    def __str__(self) -> str:   
        return "Robot at position: " + str(self.get_latest_pose())

    def show(self):
        plt.figure()
        plt.plot(self._x, self._y)
        plt.show(block=False)
