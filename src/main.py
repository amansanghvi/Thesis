from gridmap import GridMap
from imu import IMU
from lidar import Lidar


if __name__ == "__main__":
    lidar_data = Lidar("../data/lidar")
    imu_data = IMU("../data/imu", "../data/speed")
    map = GridMap()

    print(lidar_data)
    print(imu_data)
    print(map)
    # lidarData.show_all()
