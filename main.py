import matlab.engine
import matplotlib.pyplot as plt
import numpy as np

from DefaultIMUData import DefaultIMUData
from DefaultLidarData import DefaultLidarData
from gridmap import GridMap
from imu import IMU
from lidar import Lidar
from models import timestamp_to_time
from robot import Robot

if __name__ == "__main__":
    eng = matlab.engine.connect_matlab()

    lidar_data = Lidar(DefaultLidarData(), eng)
    imu_data = IMU(DefaultIMUData())
    map = GridMap(eng, 20, 0.1)
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

    plt.ion()

    fig, ax = plt.subplots()
    sc = ax.scatter([], [], s=2)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plotFrameNumber = 0
    
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
            if (lidar_idx < 20):
                robot._map.update(robot.get_latest_pose(), lidar_data[lidar_idx])
            else:
                robot.map_update(lidar_data[lidar_idx])

            ax.set_title("Frame: " + str(plotFrameNumber))
            plotFrameNumber += 1
            plot_x, plot_y = robot._map.get_occupied_points()
            sc.set_offsets(np.c_[plot_x, plot_y])
            fig.canvas.draw_idle()
            plt.pause(0.001)

    map.show()
    plt.show()
    print("ENDED")
    ax.set_title("COMPLETED")
    fig.canvas.draw_idle()
    plt.pause(6000)
    # lidar_data.show_all()
