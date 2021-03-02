from typing import Any, Callable
from math import pi

def timestamp_to_time(timestamp: int) -> float:
    return 0.0001*timestamp

def time_to_timestamp(time: float) -> int:
    return round(1e4*time)

class Position:
    x = 0
    y = 0
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

class Pose:
    _x = 0.0
    _y = 0.0
    _theta = 0.0
    def __init__(self, x: float, y: float, theta: float):
        self._x = x
        self._y = y
        self._theta = theta
    
    def x(self) -> float:
        return self._x

    def y(self) -> float:
        return self._y

    def theta(self) -> float:
        return self._theta

    def __str__(self):
        return "Pose: (" + str(self._x)  + ", " + str(self._y) + ") at " + str(self._theta*180/pi) + " degrees"

    def pos(self) -> Position:
        return Position(self._x, self._y)

class Reading:
    _dt = 0.0
    def __init__(self, data, timestamp: int, progress_fnc: Callable[[Pose, Any], Pose]):
        self._progress_fnc = progress_fnc
        self._timestamp = timestamp
        self._data = data

    def dt(self) -> float:
        return self._dt

    def set_dt(self, dt: float):
        self._dt = dt

    def timestamp(self) -> int:
        return self._timestamp
    
    def get_data(self):
        return self._data

    def get_moved_pose(self, pose: Pose) -> Pose:
        return self._progress_fnc(pose, self)
