from gridmap import GridMap
from imu import IMU
from lidar import Lidar


if __name__ == "__main__":
    lidar_data = Lidar("lidar")
    imu_data = IMU("imu", "speed")
    map = GridMap()

    print(lidar_data)
    print(imu_data)
    print(map)
    # lidarData.show_all()
