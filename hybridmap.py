from math import ceil, floor
from os import sched_get_priority_max
from gridmap import GridMap
from typing import Any, List, Optional, Tuple, Union, cast

import matlab
import copy
import matplotlib.pyplot as plt
import numpy as np
import sys

from lidar import Scan
from models import Pose, Position

MAP_LENGTH = 10 # metres
CELLS_PER_ROW = 100
CELL_SIZE = MAP_LENGTH / CELLS_PER_ROW
RELEVANT_POINT_DIST = 10.0
OCCUPIED_POINT_THRESHOLD = 1.0

VALID_DIST_THRESHOLD = 11.0

class HybridMapEntry:
    _min_x = 0.0
    _min_y = 0.0
    _max_x = 0.0
    _max_y = 0.0
    _map: GridMap
    _centre: Position
    _matlab: Any = None
    _map_len_m = 0
    _cell_size = 0.0
    def __init__(self, matlab, centre: Position, map_len_m: int, cell_size: float) -> None:
        self._map_len_m = map_len_m
        self._cell_size = cell_size
        self._centre = centre
        if (matlab != None):
            self._map = GridMap(matlab, map_len_m, cell_size)
            self._matlab = matlab
            map_radius = map_len_m/2

            self._min_x = centre.x - map_radius
            self._min_y = centre.y - map_radius
            self._max_x = centre.x + map_radius
            self._max_y = centre.y + map_radius

    def is_in_map(self, pos: Position) -> bool:
        return pos.x >= self._min_x and pos.x < self._max_x and pos.y >= self._min_y and pos.y < self._max_y
    
    def map(self) -> GridMap:
        return self._map
    
    def centre(self) -> Position:
        return self._centre
    
    def to_map_coords(self, point: Position) -> Position:
        return Position(point.x - self._centre.x, point.y - self._centre.y)
    
    def copy(self):
        new_entry = HybridMapEntry(None, Position(self.centre().x, self.centre().y), self._map_len_m, self._cell_size)
        new_entry._matlab = self._matlab
        new_entry._min_x, new_entry._min_y, new_entry._max_x, new_entry._max_y = self._min_x, self._min_y, self._max_x, self._max_y
        self._map = new_entry._map.copy()
class HybridMap:
    _cell_size = 0.05
    _map_len_m = 40
    _maps: List[HybridMapEntry] = []
    def __init__(self, matlab):
        self._matlab = matlab
        self._maps += [HybridMapEntry(matlab, Position(0, 0), self._map_len_m, self._cell_size)]

    def __str__(self) -> str:
        return "Hybrid Map: " + str(len(self._maps)) + " maps"
    
    def get_pr_at(self, pos: Position) -> Optional[float]:
        for map in self._maps:
            if (map.is_in_map(pos)):
                rel_pos = Position(pos.x - map.centre().x, pos.y - map.centre().y)
                cell = map.map().get_cell(rel_pos.x, rel_pos.y)
                if cell == None:
                    return None
                odds = np.exp(map._map[cast(Position, cell).x][cast(Position, cell).y])
                return odds/(1 + odds)
        return None
    
    def update(self, robot_pose: Pose, scan: Scan) -> Any:
        print("Updating hybrid map")
        global_scan = scan.from_global_reference(robot_pose)
        m = self.get_map_with_pos(robot_pose.pos(), None)
        if (m == None):
            return self
        m = cast(HybridMapEntry, m)
        # start_cell = m.map().get_cell(robot_pose.x() - m.centre().x, robot_pose.y() - m.centre().y)
        # if (start_cell == None):
        #     return self
        # start_cell = cast(Position, start_cell)
        # start_cell = Position(start_cell.x + int(m.centre().x/self._cell_size), start_cell.y + int(m.centre().y/self._cell_size))
        start_cell = Position(int(robot_pose.x()/self._cell_size), int(robot_pose.y()/self._cell_size))
        for i in range(0, len(global_scan)):
            end_is_occ = True
            dist = np.sqrt(scan[i].x**2 + scan[i].y**2)
            end_cell = Position(int(global_scan[i].x/self._cell_size), int(global_scan[i].y/self._cell_size))
            if (dist > 20):
                scale = 20.0/dist
                end_cell = Position(
                    int(start_cell.x + scale*(end_cell.x - start_cell.x)),
                    int(start_cell.y + scale*(end_cell.y - start_cell.y))
                )
                end_is_occ = False

            points_to_update = HybridMap.get_affected_points(
                start_cell.x, start_cell.y, 
                end_cell.x, end_cell.y
            )

            # TODO: Only process valid scan points (distance is less than a threshold.)

            for j, indices in enumerate(points_to_update):
                pos = Position(indices[0]*self._cell_size, indices[1]*self._cell_size)
                m = self.get_map_with_pos(pos.x, pos.y)
                if (m == None):
                    centre_x = ceil((pos.x + 1e-12 - self._map_len_m/2.0)/self._map_len_m)*self._map_len_m
                    centre_y = ceil((pos.y + 1e-12  - self._map_len_m/2.0)/self._map_len_m)*self._map_len_m
                    new_map_centre = Position(centre_x, centre_y)
                    print("Adding gridmap at:", new_map_centre, " to hybrid map for point:", pos, " Total maps: " + str(len(self._maps) + 1))
                    m = HybridMapEntry(self._matlab, new_map_centre, self._map_len_m, self._cell_size)
                    self._maps.append(m)
                m = cast(HybridMapEntry, m)

                rel_pos = Position(pos.x - m.centre().x, pos.y - m.centre().y)
                if end_is_occ and indices[0] == end_cell.x and indices[1] == end_cell.y:
                    m.map().set_occupied_pos(rel_pos.x, rel_pos.y)
                    # TODO: Update nearby point
                    pass
                else:
                    m.map().set_empty_pos(rel_pos.x, rel_pos.y)
        return self

    def get_scan_match(self, rel_scan: Scan, prev_scan: Scan, guess: Pose, pose_range: np.ndarray) -> Tuple[List[float], List[List[float]], float]:
        glob_scan = rel_scan.from_global_reference(guess)

        points_maps: List[HybridMapEntry] = []
        curr_points: List[Tuple[float, float]] = []
        ref_points:  List[Tuple[float, float]] = []
        
        # Get the limits of points to check.
        # min_x, min_y, max_x, max_y = 0.0, 0.0, 0.0, 0.0
        for i in range(0, len(glob_scan)):
            dist = np.sqrt((rel_scan.x()[i])**2 + (rel_scan.y()[i])**2)
            if (dist < VALID_DIST_THRESHOLD and dist > 1e-3):
                x, y = glob_scan.x()[i], glob_scan.y()[i]
                # if (x < min_x):
                #     min_x = x
                # if (y < min_y):
                #     min_y = y
                # if (x > max_x):
                #     max_x = x
                # if (y > max_y):
                #     max_y = y
                for m in self._maps:
                    if (m.is_in_map(Position(x, y))):
                        cell = m.map().get_cell(x - m.centre().x, y - m.centre().y)
                        if (cell == None):
                            continue
                        cell = cast(Position, cell)
                        px = m.map().index_to_distance(cell.x) + m.centre().x
                        py = m.map().index_to_distance(cell.y) + m.centre().y
                        points_maps.append(m)
                        curr_points.append((px, py))
        # min_x, min_y, max_x, max_y = min_x - 0.5, min_y -0.5, max_x + 0.5, max_y + 0.5

        # included_maps: List[HybridMapEntry] = self._maps
        # # Get a small list of maps to check.
        # for x in range(floor(min_x), ceil(max_x), self._map_len_m):
        #     for y in range(floor(min_y), ceil(max_y), self._map_len_m):
        #         map_to_add = self.get_map_with_pos(Position(x, y), None)
        #         if (map_to_add != None):
        #             map_to_add = cast(HybridMapEntry, map_to_add)
        #             if(len([m for m in included_maps if m.centre().x == map_to_add.centre().x and m.centre().y == map_to_add.centre().y]) == 0):
        #                 included_maps += [map_to_add]
        # map_to_add = self.get_map_with_pos(Position(max_x, max_y), None)
        # if (map_to_add != None):
        #     map_to_add = cast(HybridMapEntry, map_to_add)
        #     if(len([m for m in included_maps if m.centre().x == map_to_add.centre().x and m.centre().y == map_to_add.centre().y]) == 0):
        #         included_maps += [map_to_add]
        # cache_map: Union[HybridMapEntry, None] = None

        for i in range(len(points_maps)):
            centre_p = curr_points[i]
            mp = points_maps[i]
            cell = mp.map().get_cell(centre_p[0] - mp.centre().x, centre_p[1] - mp.centre().y)
            nearby_points: np.ndarray = []
            if cell != None:
                nearby_points = mp.map().get_nearby_occ_points(cast(Position,cell))
            else:
                print("FKKKKKKK")
                continue
            ref_points.extend([(p[0] + mp.centre().x, p[1] + mp.centre().y) for p in nearby_points])
    # for x in np.arange(-0.75, 0.76, 0.05):
            #     x = centre_p[0] + x
            #     for y in np.arange(-0.75, 0.76, 0.05):
            #         y = centre_p[1] + y
            #         # if (cache_map == None or (not cast(HybridMapEntry, cache_map).is_in_map(Position(x, y)))):
            #         #     for m in included_maps:
            #         #         if (m.is_in_map(Position(x, y))):
            #         #             cache_map = m
            #         #             break
            #         #     if (cache_map == None or (not cast(HybridMapEntry, cache_map).is_in_map(Position(x, y)))):
            #         #         cache_map = None
            #         # if (cache_map == None):
            #         #     continue
            #         cache_map = None
            #         for m in included_maps:
            #             if (m.is_in_map(Position(x, y))):
            #                 cache_map = m
            #                 break
            #         if cache_map == None:
            #             continue
            #         cache_map = cast(HybridMapEntry, cache_map)
            #         if (cache_map.map().is_occ_at(x - cache_map.centre().x, y - cache_map.centre().y)):
            #             # TODO: Figure out why some points aren't being added.
            #             cell = cache_map.map().get_cell(x - cache_map.centre().x, y - cache_map.centre().y)
            #             if (cell != None):
            #                 cell = cast(Position, cell)
            #                 add_x = cache_map.map().index_to_distance(cell.x) + cache_map.centre().x
            #                 add_y = cache_map.map().index_to_distance(cell.y) + cache_map.centre().y
            #                 ref_points.append([add_x, add_y])

        print("original ref_points len: ", len(ref_points))
        curr_adjusted_points = [[p[0] - guess.x(), p[1] - guess.y()] for p in curr_points]
        unique_ref_points = [[p[0] - guess.x(), p[1] - guess.y()] for p in np.unique(ref_points, axis=0)]

        print("unique ref_points len: ", len(unique_ref_points))
        valid_ref_points = [[p[0], p[1]] for p in unique_ref_points if np.sqrt(p[0]**2 + p[1]**2) < VALID_DIST_THRESHOLD + 0.5]
        valid_curr_points = [[p[0], p[1]] for p in curr_adjusted_points if np.sqrt(p[0]**2 + p[1]**2) < VALID_DIST_THRESHOLD]
        print("curr_points", len(valid_curr_points), " : ", valid_curr_points)
        print("ref_points", len(valid_ref_points), " : ", valid_ref_points)
        print("guess", guess.theta())
        print("resolution", int(1.0/self._cell_size))
        print("range", pose_range)
        try:
            p, cov, score = self._matlab.matchScanCustom(
                matlab.double(valid_curr_points),
                matlab.double(valid_ref_points if len(valid_ref_points) > 0 else [[]]),
                matlab.double([0.0, 0.0, 0.0]),
                int(1.0/self._cell_size), # Passing value as double
                matlab.double([pose_range[0], pose_range[1], np.pi/6]),
                nargout=3
            )
            print("Original returned pose: ", p[0])
            p[0][0] += guess.x()
            p[0][1] += guess.y()
            p[0][2] += guess.theta()
            return p[0], cov, score
        except:
            print("@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@ ERRRRRRR @@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@")
            raise
        
    def get_map_with_pos(self, x: Union[Position, float], y: Optional[float]) -> Optional[HybridMapEntry]:
        pos = Position(0, 0)
        if (isinstance(x, Position)):
            pos = x
        else:
            pos = Position(x, y)
        for m in self._maps:
            if (m.is_in_map(pos)):
                return m
        return None

    @staticmethod
    def get_affected_points(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if (dx == 0):
            return [(x0, y) for y in range(y0, y1+1)]
        if (dy == 0):
            return [(x, y0) for x in range(x0, x1+1)]
        xsign = 1 if x1 - x0 > 0 else -1
        ysign = 1 if y1 - y0 > 0 else -1

        steep = dy > dx
        if steep:
            dx, dy = dy, dx

        D = 2*dy - dx
        y = 0
        result = []
        for x in range(dx + 1):
            if (steep):
                result.append((x0 + xsign*y, y0 + ysign*x))
            else:
                result.append((x0 + xsign*x, y0 + ysign*y))
            if D >= 0:
                y += 1
                D -= 2*dx
            D += 2*dy
        return result

    def get_occupied_points(self):
        x = []
        y = []
        for map in self._maps:
            m = map.map()._map
            for i in range(0, len(m)):
                for j in range(0, len(m)):
                    if m[i][j] > OCCUPIED_POINT_THRESHOLD:
                        x.append(((i - len(m)/2)*self._cell_size + map.centre().x)/self._cell_size)
                        y.append(((j - len(m)/2)*self._cell_size + map.centre().y)/self._cell_size)
        return x, y

    def copy(self):
        new_map = HybridMap(self._matlab)
        new_map._maps = []
        for m in self._maps:
            new_map._maps += [m.copy()]
        return new_map

    def show(self):
        x, y = self.get_occupied_points()
        plt.figure()
        plt.scatter(x, y, s=2)
        plt.show(block=False)


