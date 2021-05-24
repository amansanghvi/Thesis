from math import pi, sqrt, floor
from typing import List, cast
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
DIST_THRESHOLD = 0.25
NUM_PARTICLES = 1

def resample(particles: List[Robot]) -> List[Robot]:
    weights = np.array([p.weight()[-1] for p in particles])
    print("Poses: ", [p.get_latest_pose() for p in particles])
    print("Weights: ", weights)
    avg = np.mean(weights)
    stddev = np.std(weights)
    print("Std dev", stddev)
    new_particles = particles
    if(max(weights)/min(weights) > 1e200):
        # resample
        resample_weights = np.log10(weights)
        print("Log weights: ", resample_weights)
        resample_weights[resample_weights == -np.inf] = 0
        if (min(resample_weights) < 0):
            resample_weights[resample_weights != 0] += abs(min(resample_weights))
        print("Adjusted: ", resample_weights)
        slice = sum(resample_weights)/len(resample_weights)
        new_sample_idxs: List[int] = []
        start_weight = np.random.random()*slice
        curr_sum = 0.0
        for i in range(len(resample_weights)):
            curr_sum += resample_weights[i]
            num_samples = floor((curr_sum - start_weight)/slice) - len(new_sample_idxs) + 1
            new_sample_idxs += [i]*num_samples
        
        if (len(resample_weights) != len(new_sample_idxs)):
            raise AssertionError("Incorrect number of resampled weights.")
        print("Slice:", slice, " Start: ", start_weight, " New idx:", new_sample_idxs)
        new_particles = []
        prev_i = -1
        for i in new_sample_idxs:
            if (prev_i == i):
                new_particles += [particles[i].copy()]
            else:
                new_particles += [particles[i]]
            prev_i = i
        print("Weights after: ", [p.weight()[-1] for p in new_particles])
        for p in new_particles:
            cast(Robot, p)._weight.append(1.0)
    return new_particles

if __name__ == "__main__":
    eng = matlab.engine.connect_matlab()

    lidar_data = Lidar(FreidLidarData(), eng)
    imu_data = IMU(FreidIMUData())
    map = GridMap(eng, 40, 0.1)
    particles = [Robot(eng) for _ in range(NUM_PARTICLES)]
    
    [p._map.update(p.get_latest_pose(), lidar_data[0]) for p in particles]
    [p._map.update(p.get_latest_pose(), lidar_data[0]) for p in particles]

    print(particles[0])
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
    last_updated_pose = particles[0].get_latest_pose()
    last_scan = lidar_data[0].from_global_reference(last_updated_pose)
    update_count = 0

    times = np.unique(np.concatenate((imu_data._times, lidar_data._times)))

    for t in times:
        imu_reading = imu_data[imu_idx]
        if imu_reading.timestamp() == t:
            imu_idx = min(imu_idx + 1, len(imu_data)-1)
            dt = imu_reading.timestamp() - prev_timestamp
            imu_reading.set_dt(dt)
            [p.imu_update(imu_reading) for p in particles]
            prev_timestamp = imu_reading.timestamp()

        if lidar_data.timestamp_for_idx(lidar_idx) == t:
            lidar_reading = lidar_data[lidar_idx]
            lidar_idx = min(lidar_idx + 1, len(lidar_data)-1)
            print("#################################")
            print("Frame: ", plotFrameNumber, " IMU: ", imu_idx)
            curr_pose = particles[0].get_latest_pose()
            dist = sqrt((last_updated_pose.x() - curr_pose.x())**2 + (last_updated_pose.y() - curr_pose.y())**2)
            dth = abs(last_updated_pose.theta() - curr_pose.theta())
            if (update_count < MAX_UPDATE_COUNT or (dist >= DIST_THRESHOLD or dth >= ROT_THRESHOLD)):
                weights = [p.map_update(lidar_reading, last_scan) for p in particles]
                particles = resample(particles)
                # robot._map.update(robot.get_latest_pose(), lidar_reading)
                if (dist >= DIST_THRESHOLD or dth >= ROT_THRESHOLD):
                    update_count = 0
                    last_updated_pose = curr_pose
                elif (update_count < MAX_UPDATE_COUNT):
                    update_count += 1
                last_scan = lidar_reading.from_global_reference(particles[0].get_latest_pose())
                plot_scan = last_scan
                sc_scan.set_offsets(np.c_[plot_scan.x()/particles[0]._map._cell_size, plot_scan.y()/particles[0]._map._cell_size])
                plot_x, plot_y = particles[0]._map.get_occupied_points()
                sc.set_offsets(np.c_[plot_x, plot_y])


            line.set_data(np.array(particles[0].x())/particles[0]._map._cell_size, np.array(particles[0].y())/particles[0]._map._cell_size)
            ax.set_title("Frame: " + str(plotFrameNumber))
            plotFrameNumber += 1
            fig.canvas.draw_idle()
            plt.pause(0.001)

    particles[0]._map.show()
    ax.set_title("COMPLETED")
    plt.show()
    print("ENDED")
    fig.canvas.draw_idle()
    plt.pause(6000)
    # lidar_data.show_all()
