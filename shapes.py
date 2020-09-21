import math
import numpy as np
import cv2
from cv2 import imshow as cv2_imshow
from ray import Ray
from intersection import Intersection
#from basic import dot, cross, normalize, length2, length, clamp_color, clamp, clip, reflect

RAY_T_MIN = 0.0001
RAY_T_MAX = 1.0e4
MAX_REFRACTIONS = 1

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

# color blueprints
black = np.array([0.,0.,0.])
white = np.array([1.,1.,1.])
shadow = np.array([-0.7, -0.7, -0.7])


class Material:
    def __init__(self, color, ambient, diffuse, specular, shinyness, refraction_weight=0.0):
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shinyness = shinyness
        self.refraction_weight = refraction_weight


class ShapeSet:
    def __init__(self):
        self.shapes = []
    def addShape(self, shape):
        self.shapes.append(shape)
    def intersect(self, intersection, light, filter_shape=None):
        '''
        intersection by reference, finds closest intersection if found
        '''
        doesIntersect = False
        for shape in self.shapes:
            if shape == filter_shape:
                #print('filtered out')
                continue
            if shape.intersect(intersection, light, self):
                doesIntersect = True
        return doesIntersect
    def doesIntersect(self, intersection, filter_shape=None):
        '''
        intersection by reference, finds closest intersection if found
        '''
        for shape in self.shapes:
            if shape == filter_shape:
                continue
            if isinstance(shape,Plane):
                continue
            if shape.doesIntersect(intersection):
                return True
        return False


class Plane():

    def __init__(self, position, normal, material, checkboard = True, tilesize = 10):
        self.position = position
        self.normal = normal
        self.checkboard = checkboard
        self.tilesize = tilesize
        self.material = material

    def doesIntersect(self, intersection):
        dDotN = dot(intersection.ray.direction, self.normal)
        if dDotN == 0.0:
            return False
        t = dot(self.position - intersection.ray.origin, self.normal) / dDotN
        if t <= RAY_T_MIN or t >= intersection.t:
            return False
        return True

    def intersect(self, intersection, light, shapeSet):
        dDotN = dot(intersection.ray.direction, self.normal)
        if dDotN == 0.0:
            return False
        t = dot(self.position - intersection.ray.origin, self.normal) / dDotN
        if t <= RAY_T_MIN or t >= intersection.t:
            return False

        intersection.t = t
        intersection.pShape = self
        intersection.color = self.getColor(intersection, light, shapeSet)
        return True

    def getColor(self, intersection, light, shapeSet):
        # Intersection has occured
        pt = intersection.ray.calculate(intersection.t)
        light_vector =normalize(light.center - pt)

        base_color = self.getBaseColor(pt)
        shadow_color = self.getShadow(pt, shapeSet, light_vector)

        # if spot is in shadow, no lighting required
        #if shadow_color >= 1.:
        light_color = self.getLight(pt, intersection.ray.direction, light, light_vector)
        #else:
        #    light_color = black
        if self.material.refraction_weight > 0.:
            refraction_color = self.getRefraction(intersection, pt, shapeSet, light)
        else:
            refraction_color = black
        #return base_color + light_color + refraction_color
        return shadow_color * (base_color + light_color + refraction_color)

    def getBaseColor(self, pt):
        return self.material.color

    def getRefraction(self, intersection, pt, shapeSet, light):
        # iterate over number of refractions

        reflection = reflect(intersection.ray.direction, self.normal)
        r = Ray(pt, reflection)
        i = Intersection(r, RAY_T_MAX)
        if shapeSet.intersect(i, light, self):
            return i.color
        return black

    def getShadow(self, pt, shapeSet, light_vector):
        # check for intersection with other objects
        r = Ray(pt, light_vector)
        i = Intersection(r, RAY_T_MAX)
        if shapeSet.doesIntersect(i, self):
            return 1 - np.sqrt(self.material.color) + 0.1
        else:
            return white

    def getLight(self, pt, pov, light, light_vector):
        # Phong model - init all colors
        effectiveColor = self.material.color * light.intensity
        ambientColor = effectiveColor * self.material.ambient
        diffuseColor = black
        specularColor = black

        # if < 0, then no light on surface == parallel

        lDotN = dot(light_vector, self.normal)
        if lDotN >= 0:
            diffuseColor = effectiveColor * self.material.diffuse * lDotN
            reflection = reflect(-light_vector, self.normal)
            rDotE = dot(reflection, pov)
            if rDotE > 0:
                factor = (rDotE) ** self.material.shinyness
                specularColor = light.color * self.material.specular * factor
        #else:
        #    print(light_vector, self.normal, lDotN)
        return ambientColor + diffuseColor + specularColor

class CheckBoard(Plane):
    def __init__(self, position, normal, material, tilesize=10):
        self.tilesize = tilesize
        super().__init__(position, normal, material)
    def getBaseColor(self, pt):
        point = pt.copy()
        point += (self.normal * -point)
        cond = sum(point // self.tilesize) % 2 == 0
        if cond:
            return self.material.color
        else:
            return black

class LightSet:
    def __init__(self):
        self.lights = []
    def addShape(self, light):
        self.lights.append(light)

class Light:
    def __init__(self, center, intensity):
        self.center = center
        self.intensity = intensity
        self.color = np.array([1., 1., 1.])

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def doesIntersect(self, intersection):
        origin = np.array(intersection.ray.origin - self.center, copy=True)
        direction = np.array(intersection.ray.direction, copy=True)

        a = sum(direction ** 2)      
        b = 2 * dot(direction, origin)
        c = sum(origin ** 2) - self.radius ** 2

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0.0:
            return False

        t1 = (-b - math.sqrt(discriminant)) / (2 * a)
        t2 = (-b + math.sqrt(discriminant)) / (2 * a)

        if t1 > RAY_T_MIN and t1 < intersection.t:
            return True
        elif t2 > RAY_T_MIN and t2 < intersection.t:
            return True
        else:
            return False
        return False

    def intersect(self, intersection, light, shapeSet):
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
        intersection.color = self.getColor(intersection, light, shapeSet)

        return True

    def getColor(self, intersection, light, shapeSet):
        # Intersection has occured
        pt = intersection.ray.calculate(intersection.t)
        light_vector =normalize(light.center - pt)

        #base_color = self.getBaseColor(pt)
        shadow_color = self.getShadow(pt, shapeSet, light_vector)

        # if spot is in shadow, no lighting required
        #if shadow_color >= 1:
        light_color = self.getLight(pt, intersection.ray.direction, light, light_vector)
        #else:
        #    light_color = black
        if self.material.refraction_weight > 0:
            refraction_color = self.getRefraction(intersection, pt, light, shapeSet)
        else:
            refraction_color = black

        #print(shadow_color, light_color, base_color)

        final_color = shadow_color * (light_color * (1 - self.material.refraction_weight) + refraction_color * self.material.refraction_weight)

        return final_color
        #  + base_color

    def getRefraction(self, intersection, pt, light, shapeSet):
        # iterate over number of refractions
        reflection = reflect(intersection.ray.direction, self.normalFromPt(pt))

        r = Ray(pt, reflection)
        i = Intersection(r, RAY_T_MAX)
        if shapeSet.intersect(i, light, self):
            return i.color
        return black


    def normalFromPt(self, point):
        return normalize(point - self.center)

    def getBaseColor(self, point):
        return self.material.color

    def getShadow(self, pt, shapeSet, light_vector):
        # check for intersection with other objects
        r = Ray(pt, light_vector)
        i = Intersection(r, RAY_T_MAX)
        if shapeSet.doesIntersect(i, self):
            return 1 - np.sqrt(self.material.color) + 0.1
        else:
            return white

    def getLight(self, pt, pov, light, light_vector):
        # Phong model - init all colors
        effectiveColor = self.material.color * light.intensity
        ambientColor = effectiveColor * self.material.ambient
        diffuseColor = black
        specularColor = black
        normal = self.normalFromPt(pt)

        # if < 0, then no light on surface == parallel
        lDotN = dot(light_vector, normal)
        if lDotN >= 0:
            diffuseColor = effectiveColor * self.material.diffuse * lDotN
            reflection = reflect(-light_vector, normal)
            rDotE = dot(reflection, pov)
            if rDotE > 0:
                factor = (rDotE) ** self.material.shinyness
                specularColor = light.color * self.material.specular * factor
        return ambientColor + diffuseColor + specularColor
