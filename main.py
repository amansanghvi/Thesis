from math import pi, sqrt, floor
from typing import List, cast
import shelve
import matlab.engine
import matplotlib.pyplot as plt
import numpy as np

from DefaultIMUData import DefaultIMUData
from DefaultLidarData import DefaultLidarData

from FreidIMUData import FreidIMUData
from FreidLidarData import FreidLidarData

from FreidCorrectIMUData import FreidCorrectIMUData
from FreidCorrectLidarData import FreidCorrectLidarData

from Freid101IMUData import Freid101IMUData
from Freid101LidarData import Freid101LidarData

from IntelIMUData import IntelIMUData
from IntelLidarData import IntelLidarData

from IntelRawIMUData import IntelRawIMUData
from IntelRawLidarData import IntelRawLidarData

from AcesIMUData import AcesIMUData
from AcesLidarData import AcesLidarData

from OberoIMUData import OberoIMUData
from OberoLidarData import OberoLidarData

from BeleIMUData import BeleIMUData
from BeleLidarData import BeleLidarData

from gridmap import GridMap
from imu import IMU
from lidar import Lidar
from models import Pose, timestamp_to_time
from robot import Robot

MAX_UPDATE_COUNT = 2
ROT_THRESHOLD = pi/9
DIST_THRESHOLD = 0.33
NUM_PARTICLES = 1

def resample(particles: List[Robot]) -> List[Robot]:
    weights = np.array([p.weight()[-1] for p in particles])
    # print("Weights: ", weights)
    new_particles = particles
    if(max(weights) - min(weights) > 200):
        # resample
        resample_weights = weights
        resample_weights[resample_weights == -np.inf] = 0
        if (min(resample_weights) < 0):
            resample_weights[resample_weights != 0] += abs(min(resample_weights))
        # print("Adjusted: ", resample_weights)
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

    lidar_data = Lidar(Freid101LidarData(), eng)
    imu_data = IMU(Freid101IMUData())
    map = GridMap(eng, 40, 0.1)
    particles = [Robot(eng) for _ in range(NUM_PARTICLES)]
    
    # [p._map.update(p.get_latest_pose(), lidar_data[0]) for p in particles]
    # [p._map.update(p.get_latest_pose(), lidar_data[0]) for p in particles]

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
    plotFrameNumber = 1550
    last_updated_pose = particles[0].get_latest_pose()
    last_scan = lidar_data[0].from_global_reference(last_updated_pose)
    update_count = 0

    times = np.unique(np.concatenate((imu_data._times, lidar_data._times)))
    t_idx = 3422

    with shelve.open('pickle/' + str(plotFrameNumber) + '.state', 'r') as shelf:
        prev_timestamp = shelf["prev_timestamp"]
        imu_idx = shelf["imu_idx"]
        lidar_idx = shelf["lidar_idx"]
        last_updated_pose = shelf["last_updated_pose"]
        last_scan = shelf["last_scan"]
        update_count = shelf["update_count"]
        robot = shelf["robot"]
        robot._map._matlab = eng
        for m in robot._map._maps:
            m._matlab = eng
            m._map._matlab = eng
        particles = [robot]
        # t_idx = times.tolist().index(shelf["t"])
        print("timestamp", prev_timestamp)
        print("imu_idx", imu_idx)
        print("lidar_idx", lidar_idx)
        print("last_updated_pose", last_updated_pose)
        print("last_scan", last_scan)
        print("update_count", update_count)

    for t in times[t_idx:]:
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
            # print("#################################")
            print("Frame: ", plotFrameNumber, " IMU: ", imu_idx)
            curr_pose = particles[0].get_latest_pose()
            dist = sqrt((last_updated_pose.x() - curr_pose.x())**2 + (last_updated_pose.y() - curr_pose.y())**2)
            rot = abs(last_updated_pose.theta() - curr_pose.theta())
            if (update_count < MAX_UPDATE_COUNT or (dist >= DIST_THRESHOLD or rot >= ROT_THRESHOLD)):
                if (plotFrameNumber % 5 < 2):
                    weights = [p.map_update(lidar_reading, last_scan, False) for p in particles]
                else:
                    weights = [p.map_update(lidar_reading, last_scan, True) for p in particles]
                particles = resample(particles)
                # robot._map.update(robot.get_latest_pose(), lidar_reading)
                if (dist >= DIST_THRESHOLD or rot >= ROT_THRESHOLD):
                    update_count = 0
                    last_updated_pose = curr_pose
                elif (update_count < MAX_UPDATE_COUNT):
                    update_count += 1
                if plotFrameNumber % 5 == 0:
                    last_scan = lidar_reading.from_global_reference(particles[0].get_latest_pose())
                    plot_scan = last_scan
                    sc_scan.set_offsets(np.c_[plot_scan.x()/particles[0]._map._cell_size, plot_scan.y()/particles[0]._map._cell_size])
                    plot_x, plot_y = particles[0]._map.get_occupied_points()
                    sc.set_offsets(np.c_[plot_x, plot_y])
                    line.set_data(np.array(particles[0].x())/particles[0]._map._cell_size, np.array(particles[0].y())/particles[0]._map._cell_size)
                else:
                    sc_scan.set_offsets(np.c_[[0], [0]])
                    plot_x, plot_y = particles[0]._map.get_occupied_points()
                    sc.set_offsets(np.c_[plot_x, plot_y])
                    line.set_data([], [])
            ax.set_title("Frame: " + str(plotFrameNumber))
            plotFrameNumber += 1
            fig.canvas.draw_idle()
            plt.pause(0.001)
            if (plotFrameNumber % 50 == 0):
                print("timestamp", prev_timestamp)
                print("imu_idx", imu_idx)
                print("lidar_idx", lidar_idx)
                print("last_updated_pose", last_updated_pose)
                print("last_scan", last_scan)
                print("update_count", update_count)
                print("t", t, times.tolist().index(t))
                shelf = shelve.open("pickle/" + str(plotFrameNumber) + ".state")
                shelf["prev_timestamp"] = prev_timestamp
                shelf["imu_idx"] = imu_idx
                shelf["lidar_idx"] = lidar_idx
                shelf["last_updated_pose"] = last_updated_pose
                shelf["last_scan"] = last_scan
                shelf["update_count"] = update_count
                shelf["plotFrameNumber"] = plotFrameNumber
                shelf["t"] = t
                robot = particles[0]
                robot._map._matlab = None
                for m in robot._map._maps:
                    m._matlab = None
                    m._map._matlab = None
                shelf["robot"] = particles[0]
                shelf.close()
                robot._map._matlab = eng
                for m in robot._map._maps:
                    m._matlab = eng
                    m._map._matlab = eng

                particles[0]._map.show()
                fig.canvas.draw_idle()
                

    particles[0]._map.show()
    ax.set_title("COMPLETED")
    plt.show()
    print("ENDED")
    fig.canvas.draw_idle()
    plt.pause(6000)
    # lidar_data.show_all()
