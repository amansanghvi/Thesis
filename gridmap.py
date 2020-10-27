from lidar import Scan
import numpy as np

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
    
    def update(self, scan: Scan):
        # Temporary
        scan.x()
    
    def get_cell(self, x: float, y: float): # Global x and y in metres
        if y < -self._size/2 and y > self._size/2:
            return None
        elif x < -self._size/2 and x > self._size/2:
            return None
        return x/self._size * len(self._map)


