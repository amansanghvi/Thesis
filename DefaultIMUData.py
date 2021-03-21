from models import Pose, Reading
from scipy.io import loadmat
import numpy as np
from IMUData import IMUData
from math import sin, cos, pi

NUM_REF_POINTS = 1000
class DefaultIMUData(IMUData):
    def load_and_format(self):
        imu_data = loadmat("./data/imu")
        encoder_data = loadmat("./data/speed")

        raw_omega_data = imu_data["IMU"]['DATAf'][0][0][5]
        baseline_omega = sum(raw_omega_data[0:NUM_REF_POINTS])/NUM_REF_POINTS

        raw_speed_data = encoder_data["Vel"]['speeds'][0][0][0]
        baseline_speed = sum(raw_speed_data[0:NUM_REF_POINTS])/NUM_REF_POINTS

        omega = np.array([x - baseline_omega for x in raw_omega_data])
        speed = np.array([x - baseline_speed for x in raw_speed_data])
        times = np.array([x for x in imu_data["IMU"]['times'][0][0][0]])
        
        return np.vstack((speed, omega)).transpose(), times
    
    @staticmethod
    def progress_pose(prev_pose: Pose, reading: Reading) -> Pose:
        theta = prev_pose.theta() + reading.dt()*reading.get_data()[1]
        x = prev_pose.x() + reading.dt() * reading.get_data()[0] * cos(theta)
        y = prev_pose.y() + reading.dt() * reading.get_data()[0] * sin(theta)

        return Pose(x, y, theta)

    @staticmethod
    def get_cov_change_matrix(prev_pose: Pose, reading: Reading) -> np.ndarray:
        result = np.diag([1.0, 1.0, 1.0])
        # 1.1 is just extra noise from the speed reading
        result[0][2] = reading.dt() * reading.get_data()[0] * cos(prev_pose.theta())
        result[1][2] = reading.dt() * reading.get_data()[0] * sin(prev_pose.theta())
        return result
    
    @staticmethod
    def get_cov_input_uncertainty(prev_pose: Pose, reading: Reading) -> np.ndarray:
        sensor_uncertainty = np.array([
            [reading.dt()*cos(prev_pose.theta()), 0], 
            [reading.dt()*sin(prev_pose.theta()), 0], 
            [0, reading.dt()]
        ], dtype=np.longdouble)
        magnitudes = np.diag([0.05**2, (pi/180/2)**2])
        output_uncertainty_component = np.abs(np.matmul(np.matmul(sensor_uncertainty, magnitudes), sensor_uncertainty.transpose()))
        noise_uncertainty_component = np.abs(np.diag([(0.01)**2, (0.01)**2, (0.2*pi/180)**2]))
        return output_uncertainty_component + noise_uncertainty_component
