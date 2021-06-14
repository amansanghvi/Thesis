from abc import abstractclassmethod
from typing import Any, List, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np

from lidar import Scan
from models import Pose, Position

MAP_LENGTH = 10 # metres
CELLS_PER_ROW = 100
CELL_SIZE = MAP_LENGTH / CELLS_PER_ROW
RELEVANT_POINT_DIST = 10.0
OCCUPIED_POINT_THRESHOLD = 1.0

class Map:
    
    @abstractclassmethod
    def __getitem__(self, idx: int) -> list: # Hacky way to allow double indexing
        pass
    @abstractclassmethod
    def __len__(self) -> int:
        pass

    @abstractclassmethod
    def __str__(self) -> str:
        pass
    
    @abstractclassmethod
    def get_pr_at(self, pos: Position) -> Optional[float]:
        pass
    
    @abstractclassmethod
    def update(self, robot_pose: Pose, scan: Scan) -> Any:
        pass

    # Input is GLOBAL x and y in metres
    @abstractclassmethod
    def get_cell(self, x: float, y: float) -> Optional[Position]:
        pass
    
    @abstractclassmethod
    def get_nearby_occ_points(self, curr_cell: Position) -> np.ndarray:
        pass
    
    @abstractclassmethod
    def get_scan_match(self, rel_scan: Scan, prev_scan: Scan, guess: Pose, pose_range: np.ndarray) -> Tuple[List[float], List[List[float]], float]:
        pass
    
    @abstractclassmethod
    def is_occ_at(self, x, y) -> bool:
        pass

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
    # Uses an arbitrary unit of distance.
    @abstractclassmethod
    def get_occupied_points(self):
        pass
    
    # index to m from origin.
    @abstractclassmethod
    def index_to_distance(self, i: int) -> float:
        pass

    @abstractclassmethod
    def copy(self) -> Any:
        pass

    def show(self):
        x, y = self.get_occupied_points()
        plt.figure()
        plt.scatter(x, y, s=2)
        plt.show(block=False)


