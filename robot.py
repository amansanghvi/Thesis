from math import pi
from typing import List, cast
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scipy
import copy

from gridmap import GridMap
from hybridmap import HybridMap
from imu import Reading
from lidar import Scan
from models import Pose

MAP_LENGTH = 10 # metres
CELLS_IN_ROW = 100
CELL_SIZE = MAP_LENGTH / CELLS_IN_ROW
NUM_SAMPLE_POINTS = 30

class Robot:
    def __init__(self, matlab):
        if (matlab != None):
            # self._map = GridMap(matlab, 50, 0.05)
            self._map = HybridMap(matlab)
            self._weight = [1.0]
            self._cov = np.zeros((3, 3), dtype=np.longdouble)
            self._x = [0.0]
            self._y = [0.0]
            self._theta = [0.0]

    def x(self) -> np.ndarray:
        return self._x

    def y(self) -> np.ndarray:
        return self._y

    def theta(self) -> np.ndarray:
        return self._theta
    
    def weight(self) -> np.ndarray:
        return self._weight

    def get_latest_pose(self) -> Pose:
        return Pose(self._x[-1], self._y[-1], self._theta[-1])

    def imu_update(self, reading: Reading) -> Pose:
        prev_pose = self.get_latest_pose()
        next_pose = reading.get_moved_pose(prev_pose)
        change_matrix = reading.get_cov_change_matrix(prev_pose)

        self._cov = np.matmul(np.matmul(change_matrix, self._cov), change_matrix.transpose())
        self._cov += reading.get_cov_input_uncertainty(prev_pose)

        self._x.append(next_pose.x())
        self._y.append(next_pose.y())
        self._theta.append(next_pose.theta())

        return next_pose

    def map_update(self, scan: Scan, last_scan: Scan, adj: bool):
        latest_pose = self.get_latest_pose()
        # We only search within a restricted range 3.0 standard deviations of the mean.
        pose_range = np.sqrt(np.diag(self._cov))*30.0
        pose_range[2] = max(min(4*pose_range[2], pi/3), pi/8)
        pose_range[1] = max(min(4*pose_range[1], 0.7), 0.1)
        pose_range[0] = max(min(4*pose_range[0], 0.7), 0.1)
        if adj:
            scan_pose, scan_cov, score = self._map.get_scan_adj(scan, last_scan, latest_pose, pose_range)
        else:
            scan_pose, scan_cov, score = self._map.get_scan_match(scan, last_scan, latest_pose, pose_range)
        # print("poses: ", scan_pose, " vs ", latest_pose)
        # print("SCORE: ", score)

        if (np.isnan(scan_cov).any()):
            print("BAD SCORE")
            self._map.update(latest_pose, scan)
            weight_update = self._generate_sample_weight([[latest_pose.x(), latest_pose.y(), latest_pose.theta()]], scan, [1])
            self._weight.append(weight_update[0] + self._weight[-1])
            return

        K_sample_points = NUM_SAMPLE_POINTS
        guesses = np.random.multivariate_normal(scan_pose, np.array(scan_cov), K_sample_points)

        # TODO: Workout when to use scan match and odometry.
        predicted_odd_mean = scan_pose # [latest_pose.x(), latest_pose.y(), latest_pose.theta()]
        predicted_odd_cov = self._cov

        motion_prs = scipy.multivariate_normal.pdf(guesses, scan_pose, scan_cov)*10
        ksample_weights = self._generate_sample_weight(guesses, scan, motion_prs)
        min_w = min(ksample_weights)
        ksample_weights = [k - min_w + 1e-2 for k in ksample_weights]

        mean = np.zeros(3, dtype=np.longdouble)
        sigma = np.zeros((3, 3), dtype=np.longdouble)
        norm = np.longdouble(0.0)
        for i in range(len(guesses)):
            mean = np.add(mean, guesses[i] * ksample_weights[i])
            norm = norm + ksample_weights[i]

        if (abs(norm) < -1):
            mean = predicted_odd_mean
            sigma = predicted_odd_cov
        else:
            mean = mean/norm
            for i in range(len(guesses)):
                delta_pos = np.matrix(np.add(guesses[i], -mean))
                sigma = sigma + delta_pos.T*delta_pos*ksample_weights[i]
            sigma = np.array(sigma)/norm
            norm = norm + min_w * len(ksample_weights)
        mean_pose = Pose(mean[0], mean[1], mean[2])
        self._cov = sigma
        self._x.append(mean_pose.x())
        self._y.append(mean_pose.y())
        self._theta.append(mean_pose.theta())
        self._weight.append(norm+self._weight[-1])
        self._map.update(mean_pose, scan)

    
    def _generate_sample_weight(self, guesses: np.ndarray, scan: Scan, motion_prs: np.ndarray) -> List[float]:
        ksample_weights = np.zeros(len(guesses), dtype=np.longdouble)
        for i in range(len(guesses)):
            x_k = guesses[i]
            position_weight = motion_prs[i]  # motion_model.pdf(x_k)
            
            observation_weight = np.longdouble(1.0)
            multiplier = 1
            adjusted_scan = scan.from_global_reference(Pose(x_k[0], x_k[1], x_k[2]))
            for j in range(len(adjusted_scan)):
                beam = adjusted_scan[j]
                dist = np.sqrt(scan[j].x**2 + scan[j].y**2)
                if dist < 25 and dist > 0.01: # If not out of range
                    pr_occ = self._map.get_odds_at(beam)
                    if (pr_occ == None):
                        observation_weight += 0
                    else:
                        observation_weight += cast(float, pr_occ)*multiplier # Arbitrary scaling applied (*10)
                else:
                    observation_weight += 0
            ksample_weights[i] = observation_weight * position_weight
        return ksample_weights

    def copy(self):
        new_robot = Robot(None)
        new_robot._weight = self._weight.copy()
        new_robot._x = self._x.copy()
        new_robot._y = self._y.copy()
        new_robot._theta = self._theta.copy()
        new_robot._cov = copy.deepcopy(self._cov)
        new_robot._map = self._map.copy()
        return new_robot

    def __str__(self) -> str:   
        return "Robot at position: " + str(self.get_latest_pose())

    def show(self):
        plt.figure()
        plt.plot(self._x, self._y)
        plt.show(block=False)
