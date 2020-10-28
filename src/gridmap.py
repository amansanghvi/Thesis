from robot import Robot
from lidar import Position, Scan
import numpy as np
import matplotlib.pyplot as plt

MAP_LENGTH = 10 # metres
CELLS_PER_ROW = 100
CELL_SIZE = MAP_LENGTH / CELLS_PER_ROW

class GridMap:
    _map = [[]]
    _size = 0 # in metres
    def __init__(self, map_len=MAP_LENGTH, cell_size=CELL_SIZE):
        if map_len < 1:
            raise Exception("Cannot have map length less than 1m")
        dim = round(map_len/cell_size)
        self._map = np.zeros((dim, dim))
        self._map.fill(False)
        self._size = map_len
    
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
            if cell != None:
                self._map[round(cell.x), round(cell.y)] = True
        return self
    
    def get_cell(self, x: float, y: float) -> Position: # Global x and y in metres
        if y < -self._size/2 or y > self._size/2:
            return None
        elif x < -self._size/2 or x > self._size/2:
            return None
        return Position(x/self._size * len(self._map) + len(self._map)/2, y/self._size * len(self._map) + len(self._map)/2)
    
    def show(self):
        x = []
        y = []
        for i in range(0, len(self._map)):
            for j in range(0, len(self._map)):
                if self._map[i][j]:
                    x.append(i - len(self._map)/2)
                    y.append(j - len(self._map)/2)
        print(x)
        plt.figure()
        plt.scatter(x, y, s=2)
        plt.show(block=False)


