import math
import numpy as np
import cv2
from cv2 import imshow as cv2_imshow

from ray import Ray

RAY_T_MIN = 0.0001
RAY_T_MAX = 1.0e3

def dot(v1, v2):
    return(np.dot(v1, v2))
def cross(v1, v2):
    return np.cross(v1, v2)
def normalize(v):
    return v / sum(abs(v))
def length2(v1, v2):
    return (v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2
def length(f):
    return math.sqrt(f)
def clamp_color(v):
    return np.array([clamp(v[0]), clamp(v[1]), clamp(v[2])])
def clamp(f):
    return 255 if f > 255 else (0 if f < 0 else int(f))
def clip(x, min_x, max_x):
    if min_x < x:
        return min_x
    elif max_x < x:
        return max_x
    return x
def reflect(incoming, normal):
    return incoming - normal * 2. * dot(incoming, normal)

class Image:
    def __init__(self):
        pass
    def create_image(self, H, W):
        self.H = H
        self.W = W
        self.image = np.zeros((self.H, self.W, 3))
    def get_image(self):
        return np.flip(np.flip(self.image, axis=0), axis=1)


class Camera:
    def __init__(self, vertical_resolution, origin, direction, up, right, fov, ratio):

        self.H = int(vertical_resolution / 2)
        self.H_it = self.H * 2
        self.W = int(self.H * ratio)
        self.W_it = self.W * 2
        self.origin = origin
        self.direction = direction
        self.up = up
        self.right = right
        self.increment = fov / self.H
        self.calculateRayMap()

    def calculateRayMap(self):
        self.RayMap = {}
        for y in range(-self.H, self.H + 1):
            horizontal_add = math.tan(self.increment * y * math.pi / 180)
            for x in range(-self.W, self.W + 1):
                vertical_add = math.tan(self.increment * x * math.pi / 180)
                target_point = normalize((vertical_add * self.right + self.direction) + (horizontal_add * self.up + self.direction))
                self.RayMap[(y + self.H,x + self.W)] = Ray(self.origin , target_point)
                
    def getRay(self, x, y):
        return self.RayMap[(x,y)]