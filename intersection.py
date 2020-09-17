import math
import numpy as np
import cv2
from cv2 import imshow as cv2_imshow


RAY_T_MIN = 0.0001
RAY_T_MAX = 1.0e3

class Intersection:
    def __init__(self,ray, t, pShape=None):
        self.ray = ray
        self.t = t
        self.pShape = pShape
        self.color = np.array([0,0,0])
    def intersected(self):
        return (self.pShape != None)
    def position(self):
        return self.ray.calculate(self.t)