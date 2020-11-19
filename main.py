from DefaultIMUData import DefaultIMUData
from DefaultLidarData import DefaultLidarData
from models import timestamp_to_time
import matlab.engine
import matplotlib.pyplot as plt
from robot import Robot
from gridmap import GridMap
from imu import IMU
from lidar import Lidar

if __name__ == "__main__":
    eng = matlab.engine.connect_matlab()

    lidar_data = Lidar(DefaultLidarData(), eng)
    imu_data = IMU(DefaultIMUData())
    map = GridMap(eng, 20)
    robot = Robot(10, eng)
    
    map.update(robot.get_latest_pose(), lidar_data[0])
    # print(map.get_scan_match(lidar_data[1800]))
    
    print(robot)
    print(lidar_data)
    print(imu_data)
    print(map)
    
    prev_timestamp = imu_data[0].timestamp()
    imu_idx = 0
    lidar_idx = 0
    
    for t in range(prev_timestamp, imu_data[-1].timestamp(), 10):
        if (t > imu_data[imu_idx].timestamp()):
            imu_idx = min(imu_idx + 1, len(imu_data)-1)
        if (t > lidar_data.timestamp_for_idx(lidar_idx)):
            lidar_idx = min(lidar_idx + 1, len(lidar_data)-1)
        
        imu_reading = imu_data[imu_idx]
        if imu_reading.timestamp() == t:
            dt = timestamp_to_time(imu_reading.timestamp() - prev_timestamp)
            imu_reading.set_dt(dt)
            robot.imu_update(imu_reading)
            prev_timestamp = imu_reading.timestamp()
        
        if lidar_data.timestamp_for_idx(lidar_idx) == t:
            map.update(robot.get_latest_pose(), lidar_data[lidar_idx])

    map.show()
    plt.show()
    # lidar_data.show_all()
