from math import pi

def timestamp_to_time(timestamp: int) -> float:
    return 0.0001*timestamp

class Reading:
    _omega = 0
    _speed = 0
    _dt = 0
    _timestamp = 0

    def __init__(self, omega: float, speed: float, timestamp: int):
        self._omega = omega
        self._speed = speed
        self._timestamp = timestamp
    
    def omega(self) -> float:
        return self._omega

    def speed(self) -> float:
        return self._speed

    def dt(self) -> float:
        return self._dt

    def set_dt(self, dt: float):
        self._dt = dt

    def timestamp(self) -> int:
        return self._timestamp

class Position:
    x = 0
    y = 0
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

class Pose:
    _x = 0
    _y = 0
    _theta = 0
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

