from math import pi, sqrt
from typing import cast
import matlab.engine
import matplotlib.pyplot as plt
import numpy as np

from DefaultIMUData import DefaultIMUData
from DefaultLidarData import DefaultLidarData

from FreidIMUData import FreidIMUData
from FreidLidarData import FreidLidarData

from IntelIMUData import IntelIMUData
from IntelLidarData import IntelLidarData

from IntelRawIMUData import IntelRawIMUData
from IntelRawLidarData import IntelRawLidarData

from gridmap import GridMap
from imu import IMU
from lidar import Lidar
from models import Pose, timestamp_to_time
from robot import Robot

MAX_UPDATE_COUNT = 2
ROT_THRESHOLD = pi/9
DIST_THRESHOLD = 0.1

if __name__ == "__main__":
    eng = matlab.engine.connect_matlab()

    lidar_data = Lidar(FreidLidarData(), eng)
    imu_data = IMU(FreidIMUData())
    map = GridMap(eng, 40, 0.1)
    robot = Robot(10, eng)
    
    robot._map.update(robot.get_latest_pose(), lidar_data[0])
    robot._map.update(robot.get_latest_pose(), lidar_data[0])
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
    sc_scan = ax.scatter([], [], s=2)
    line = ax.plot([0], [0], 'r-')[0]
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plotFrameNumber = 0
    last_updated_pose = robot.get_latest_pose()
    last_scan = lidar_data[0].from_global_reference(last_updated_pose)
    update_count = 0

    times = np.unique(np.concatenate((imu_data._times, lidar_data._times)))

    for t in times:
        imu_reading = imu_data[imu_idx]
        if imu_reading.timestamp() == t:
            imu_idx = min(imu_idx + 1, len(imu_data)-1)
            dt = imu_reading.timestamp() - prev_timestamp
            imu_reading.set_dt(dt)
            robot.imu_update(imu_reading)
            prev_timestamp = imu_reading.timestamp()

        if lidar_data.timestamp_for_idx(lidar_idx) == t:
            lidar_reading = lidar_data[lidar_idx]
            lidar_idx = min(lidar_idx + 1, len(lidar_data)-1)
            # map.update(robot.get_latest_pose(), lidar_reading)
            print("#################################")
            print("Frame: ", plotFrameNumber, " IMU: ", imu_idx)
            curr_pose = robot.get_latest_pose()
            dist = sqrt((last_updated_pose.x() - curr_pose.x())**2 + (last_updated_pose.y() - curr_pose.y())**2)
            dth = abs(last_updated_pose.theta() - curr_pose.theta())
            if (update_count < MAX_UPDATE_COUNT or (dist >= DIST_THRESHOLD or dth >= ROT_THRESHOLD)):
                robot.map_update(lidar_reading, last_scan)
                if (dist >= DIST_THRESHOLD or dth >= ROT_THRESHOLD):
                    update_count = 0
                    last_updated_pose = curr_pose
                elif (update_count < MAX_UPDATE_COUNT):
                    update_count += 1
                last_scan = lidar_reading.from_global_reference(robot.get_latest_pose())
                plot_scan = last_scan
                sc_scan.set_offsets(np.c_[plot_scan.x()/robot._map._cell_size, plot_scan.y()/robot._map._cell_size])
                plot_x, plot_y = robot._map.get_occupied_points()
                # plot_x, plot_y = map.get_occupied_points()
                sc.set_offsets(np.c_[plot_x, plot_y])

            line.set_data(np.array(robot.x())/robot._map._cell_size, np.array(robot.y())/robot._map._cell_size)
            ax.set_title("Frame: " + str(plotFrameNumber))
            plotFrameNumber += 1
            fig.canvas.draw_idle()
            plt.pause(0.001)

    robot._map.show()
    ax.set_title("COMPLETED")
    plt.show()
    print("ENDED")
    fig.canvas.draw_idle()
    plt.pause(6000)
    # lidar_data.show_all()
