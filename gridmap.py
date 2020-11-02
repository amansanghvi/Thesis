from robot import Robot
from lidar import Scan
import numpy as np
import matplotlib.pyplot as plt
from models import Pose, Position
import matlab

MAP_LENGTH = 10 # metres
CELLS_PER_ROW = 100
CELL_SIZE = MAP_LENGTH / CELLS_PER_ROW

class GridMap:
    _map = [[]]
    _size = 0 # in metres
    _matlab = None
    def __init__(self, matlab, map_len=MAP_LENGTH, cell_size=CELL_SIZE):
        if map_len < 1:
            raise Exception("Cannot have map length less than 1m")
        dim = round(map_len/cell_size)
        self._map = np.zeros((dim, dim))
        self._map.fill(False)
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
    
    def update(self, robot: Robot, scan: Scan):
        global_scan = scan.from_global_reference(robot.get_latest_pose())
        for i in range(0, len(global_scan)):
            cell = self.get_cell(global_scan[i].x, global_scan[i].y)
            if cell != None and abs(cell.x) < len(self._map) and abs(cell.y) < len(self._map):
                self._map[cell.x][cell.y] = True
        return self
    
    def get_cell(self, x: float, y: float) -> Position: # Global x and y in metres
        if y <= -self._size/2 or y >= self._size/2:
            return None
        elif x <= -self._size/2 or x >= self._size/2:
            return None
        return Position(
            round(x/self._size * len(self._map) + len(self._map)/2), 
            round(y/self._size * len(self._map) + len(self._map)/2)
        )
    
    def get_scan_match(self, scan: Scan) -> Pose: # Global x and y in metres
        ref_points = []
        for x in range(0, len(self._map)):
            for y in range(0, len(self._map)):
                if (self._map[x][y]):
                    ref_points.append([ # TODO: Think this through
                        float(x - len(self._map)/2) * self._size/len(self._map), 
                        float(y - len(self._map)/2) * self._size/len(self._map), 
                    ])
        curr_points = []
        for i in range(0, len(scan)):
            curr_points.append([scan.x()[i], scan.y()[i]])
        p = self._matlab.matchScanCustom(matlab.double(curr_points), matlab.double(ref_points))[0]
        print(p)
        return Pose(p[0], p[1], p[2])
        
    
    def show(self):
        x = []
        y = []
        for i in range(0, len(self._map)):
            for j in range(0, len(self._map)):
                if self._map[i][j]:
                    x.append(i - len(self._map)/2)
                    y.append(j - len(self._map)/2)
        plt.figure()
        plt.scatter(x, y, s=2)
        plt.show(block=False)


