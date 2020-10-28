from math import pi
import matplotlib.pyplot as plt
from robot import Robot
from gridmap import GridMap
from imu import IMU
from lidar import Lidar

if __name__ == "__main__":
    lidar_data = Lidar("./data/lidar")
    imu_data = IMU("./data/imu", "./data/speed")
    map = GridMap(20)
    robot = Robot()
    # TODO: Plot robot path
    # robot._theta = [pi/2]
    map.update(robot, lidar_data[0]).show()
    lidar_data[0].show()
    for reading in imu_data:
        robot.imu_update(reading)
    
    robot.show()
    print(robot)
    print(lidar_data)
    print(imu_data)
    print(map)
    plt.show()
    # lidar_data.show_all()
