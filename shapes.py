import math
import numpy as np
import cv2
from cv2 import imshow as cv2_imshow

#from basic import dot, cross, normalize, length2, length, clamp_color, clamp, clip, reflect

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


class Material:
    def __init__(self, color, ambient, diffuse, specular, shinyness):
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shinyness = shinyness

class ShapeSet:
    def __init__(self):
        self.shapes = []
    def addShape(self, shape):
        self.shapes.append(shape)
    def intersect(self, intersection):
        '''
        intersection by reference, finds closest intersection if found
        '''
        doesIntersect = False
        for shape in self.shapes:
            if shape.intersect(intersection):
                #print(intersection.t)
                doesIntersect = True
        return doesIntersect

    def filtered(self, shape):
        return [i for i in self.shapes if i != shape]

class Plane():
    def __init__(self, position, normal, material, checkboard = True, tilesize = 10):
        self.position = position
        self.normal = normal
        self.checkboard = checkboard
        self.tilesize = tilesize
        self.material = material

    def intersect(self, intersection):
        dDotN = dot(intersection.ray.direction, self.normal)
        if dDotN == 0.0:
            return False

        t = dot(self.position - intersection.ray.origin, self.normal) / dDotN
        if t <= RAY_T_MIN or t >= intersection.t:
            return False
        intersection.t = t
        intersection.pShape = self
        # return dynamic checkboard color
        if self.checkboard == True:
            point = intersection.ray.calculate(intersection.t)
            point += (self.normal * -point)
            cond = sum(point // self.tilesize) % 2 == 0
            if cond:
                intersection.color = np.array([255,255,255])
            else:
                intersection.color = np.array([0,0,0])
        else:
            intersection.color = self.material.color
        return True


class Light:
    def __init__(self, center, intensity):
        self.center = center
        self.intensity = intensity


class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, intersection):
        '''
        intersection passed as reference, t value filled out with smallest intersection
        '''
        origin = np.array(intersection.ray.origin - self.center, copy=True)
        direction = np.array(intersection.ray.direction, copy=True)

        a = sum(direction ** 2)      
        b = 2 * dot(direction, origin)
        c = sum(origin ** 2) - self.radius ** 2

        discriminant = b ** 2 - 4 * a * c
        # avoid division by 0
        if discriminant < 0.0:
            return False

        t1 = (-b - math.sqrt(discriminant)) / (2 * a)
        t2 = (-b + math.sqrt(discriminant)) / (2 * a)


        if t1 > RAY_T_MIN and t1 < intersection.t:
            intersection.t = t1
        elif t2 > RAY_T_MIN and t2 < intersection.t:
            intersection.t = t2
        else:
            return False
        intersection.pShape = self
        #intersection.color = self.material.color + self.lighting(intersection.ray.calculate(intersection.t))
        return True

    def normalFromPt(self, point):
        return normalize(self.center - point)

    def lighting(self, pt, light, pov):
        black = np.array([0,0,0])
        effectiveColor = self.material.color * light.intensity
        ambientColor = effectiveColor * self.material.ambient
        diffuseColor = black
        specularColor = black
        light_vector = normalize(light.center - pt)

        lDotN = dot(light_vector, self.normalFromPt(pt))
        if lDotN < 0:
            diffuseColor = black
            specularColor = black
        else:
            diffuseColor = effectiveColor * self.material.diffuse * lDotN
            reflection = reflect(-light_vector, self.normalFromPt(pt))
            rDotE = dot(reflection, pov)
            if rDotE <= 0:
                specularColor = black
            else:
                factor = rDotE ** self.material.shinyness
                specularColor = light.intensity * self.material.specular * factor

        #print(ambientColor, diffuseColor, specularColor)
        return ambientColor + diffuseColor + specularColor


m = Material(color=np.array([1,1,1]), ambient=0.1, diffuse=0.9, specular=0.9, shinyness=200.)