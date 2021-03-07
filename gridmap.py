from typing import Any, List, Optional, Tuple, cast

import matlab
import matplotlib.pyplot as plt
import numpy as np

from lidar import Scan
from models import Pose, Position

MAP_LENGTH = 10 # metres
CELLS_PER_ROW = 100
CELL_SIZE = MAP_LENGTH / CELLS_PER_ROW

class GridMap:
    _map = np.array([])
    _size = 0 # in metres
    _matlab: Any = None
    log_odds_occ = 1.0 # Around 80% chance of lidar being right about which cell.
    log_odds_nearby = 0.20
    max_odds_occ = 3.0  # Can only be at most ~95% confident on occupancy.
    log_odds_emp = -0.45  # Probability of 0.2.
    min_odds_emp = -2.2  # Can only be at most ~90% sure a cell is empty.
    def __init__(self, matlab, map_len=MAP_LENGTH, cell_size=CELL_SIZE):
        if map_len < 1:
            raise Exception("Cannot have map length less than 1m")
        dim = round(map_len/cell_size)
        self._map = np.zeros((dim, dim))
        self._map.fill(0.0)
        self._size = map_len
        self._matlab = matlab
    
    def __getitem__(self, idx: int) -> list: # Hacky way to allow double indexing
        if not isinstance(idx, int):
            raise Exception("Invalid attribute: " + str(idx))
        return self._map[idx]

    def __len__(self) -> int:
        return len(self._map)

    def __str__(self) -> str:
        return "Map: " + str(len(self._map)) + "x" + str(len(self._map[0])) + " cells"
    
    def get_pr_at(self, pos: Position):
        cell = self.get_cell(pos.x, pos.y)
        if cell == None:
            return None
        odds = np.exp(self._map[cast(Position, cell).x][cast(Position, cell).y])
        return odds/(1 + odds)
    
    def update(self, robot_pose: Pose, scan: Scan) -> Any:
        global_scan = scan.from_global_reference(robot_pose)
        start_cell = self.get_cell(robot_pose.x(), robot_pose.y())
        if start_cell == None:
            return self
        for i in range(0, len(global_scan)):
            end_cell = self.get_cell(global_scan[i].x, global_scan[i].y)
            if end_cell == None:
                continue
            end_cell = cast(Position, end_cell)
            start_cell = cast(Position, start_cell)
            points_to_update = GridMap.get_affected_points(
                start_cell.x, start_cell.y, 
                end_cell.x, end_cell.y
            )

            for j, point in enumerate(points_to_update):
                if point[0] == end_cell.x and point[1] == end_cell.y:
                    self._map[point[0]][point[1]] = min(
                        self._map[point[0]][point[1]] + self.log_odds_occ, 
                        self.max_odds_occ
                    )
                    prev_x, prev_y = points_to_update[j-1 if j > 0 else 1] # If endpoint is first for some reason, 
                    self._map[prev_x][prev_y] += self.log_odds_nearby
                else:
                    self._map[point[0]][point[1]] = max(
                        self._map[point[0]][point[1]] + self.log_odds_emp, 
                        self.min_odds_emp
                    )
        return self
    
    # Input is GLOBAL x and y in metres
    def get_cell(self, x: float, y: float) -> Optional[Position]:
        if y <= -self._size/2 or y >= self._size/2:
            return None
        elif x <= -self._size/2 or x >= self._size/2:
            return None
        return Position( 
            # Use int() to truncate because we assume cells are either completely full
            # We round the number closer to zero.
            int(x/self._size * len(self._map) + len(self._map)/2), 
            int(y/self._size * len(self._map) + len(self._map)/2)
        )
    
    def get_scan_match(self, scan: Scan, guess: Pose) -> Tuple[List[float], List[List[float]]]:
        ref_points = []
        
        for x in range(0, len(self._map)):
            for y in range(0, len(self._map)):
                if (self._map[x][y] > 0.1):
                    ref_points.append([ # TODO: Think this through
                        self.index_to_distance(x), 
                        self.index_to_distance(y)
                    ])
        curr_points = []
        for i in range(0, len(scan)):
            curr_cell = self.get_cell(scan.x()[i], scan.y()[i])
            if (curr_cell == None):
                continue
            curr_cell = cast(Position, curr_cell)
            curr_points.append([
                self.index_to_distance(curr_cell.x), 
                self.index_to_distance(curr_cell.y)
            ])
        p, cov = self._matlab.matchScanCustom(
            matlab.double(curr_points),
            matlab.double(ref_points),
            matlab.double([guess.x(), guess.y(), guess.theta()]),
            nargout=2
        )
        return p[0], cov
    
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
        
    # Does not gives accurate position. 
    # Distances between points are arbitrarily scaled.
    def get_occupied_points(self):
        x = []
        y = []
        for i in range(0, len(self._map)):
            for j in range(0, len(self._map)):
                if self._map[i][j] > 1.0:
                    x.append(i - len(self._map)/2)
                    y.append(j - len(self._map)/2) 
        return x, y
    
    def index_to_distance(self, i: int) -> float:
        return float(i - len(self._map)/2) * self._size/len(self._map)

    def show(self):
        x, y = self.get_occupied_points()
        plt.figure()
        plt.scatter(x, y, s=2)
        plt.show(block=False)


