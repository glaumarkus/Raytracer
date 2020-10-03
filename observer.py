import math
import numpy as np
import cv2
from cv2 import imshow as cv2_imshow

from ray import Ray

RAY_T_MIN = 0.0001
RAY_T_MAX = 1.0e4

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
    def __init__(self, H, aspectRatio):
        self.h = H
        self.w = int(H * aspectRatio)
        self.image = np.zeros((self.h, self.w, 3))
    def getHeight(self):
        return self.h
    def getWidth(self):
        return self.w
    def get_image(self):
        return self.image #np.flip(np.flip(self.image, axis=0), axis=1)


class Camera:
    def __init__(self, origin, target, upguide, fov, aspectRatio):

        self.origin = origin
        self.forward = normalize(target - origin)
        self.forward = target
        self.right = normalize(cross(self.forward, upguide))
        self.up = cross(self.right, self.forward)

        self.h = math.tan(fov)
        self.w = self.h * aspectRatio
        
    def getRay(self, x, y):
        direction = self.forward + x * self.w * self.right + y * self.h * self.up
        return Ray(self.origin, normalize(direction))


