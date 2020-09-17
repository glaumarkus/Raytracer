import math
import numpy as np
import cv2
from cv2 import imshow as cv2_imshow

RAY_T_MIN = 0.0001
RAY_T_MAX = 1.0e3

class Ray:
    def __init__(self, origin, direction, tMax = RAY_T_MAX):
        self.origin = origin
        self.direction = direction
        self.tMax = tMax
    def calculate(self, t):
        return self.origin + self.direction * t