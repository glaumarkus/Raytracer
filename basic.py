import math
import numpy as np
import cv2
from cv2 import imshow as cv2_imshow

from ray import Ray
from intersection import Intersection
from shapes import Material, ShapeSet, Plane, Sphere, Light
from observer import Image, Camera


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
    return 1. if f > 1. else (0. if f < 0. else f)
def clip(x, min_x, max_x):
    if min_x < x:
        return min_x
    elif max_x < x:
        return max_x
    return x
def reflect(incoming, normal):
    return incoming - normal * 2. * dot(incoming, normal)


def RayTrace(image, camera, shape, light):

    image.create_image(camera.H * 2, camera.W * 2)
    black = np.array([0.,0.,0.])
    white = np.array([1.,1.,1])
    for x in range(image.H):
        for y in range(image.W):
            if camera.getRay(x,y):
                r = camera.getRay(x,y)
                i = Intersection(r, RAY_T_MAX)
                if shape.intersect(i):
                    '''
                    local_color = i.pShape.material.color
                    light_color = black
                    if isinstance(i.pShape, Sphere):
                        light_color = i.pShape.lighting(i.ray.calculate(i.t), light, camera.origin)
                        print(local_color, light_color)
                    '''
                    if isinstance(i.pShape, Sphere):
                        local_color = i.pShape.lighting(i.ray.calculate(i.t), light, camera.origin)
                    else:
                        local_color = i.color
                    image.image[x,y] = clamp_color(local_color) #white
                else:
                    image.image[x,y] = black #np.array([0.,0.,0.])#black


'''
m = Material(color=np.array([1,1,1]), ambient=0.1, diffuse=0.9, specular=0.9, shinyness=200.)

# in seiner Function hat er 
# point = 0,0,0
# light.origin
# camera.origin
# normal = 0,0,-1

# direct
origin = np.array([0.,0.,-1])
s1 = Sphere(np.array([0,0,0]), 1, m)
r = Ray(origin, normalize(s1.center - origin))
i = Intersection(r, RAY_T_MAX)
l = Light(np.array([0,0,-10]), np.array([1,1,1]))
print('Direct Light:', s1.intersect(i))
print('Add. light:', s1.lighting(i.ray.calculate(i.t), l, origin))
#print('Normal Color:', i.pShape.material.color)
#print('Final Color:', s1.lighting(i.ray.calculate(i.t), l, origin) + i.pShape.material.color)
print('')

# off angle
origin = np.array([0., math.sqrt(2) / 2, math.sqrt(2) / -2])
r = Ray(origin, normalize(s1.center - origin))
i = Intersection(r, RAY_T_MAX)

print('Off Angle Light:', s1.intersect(i))
print('Add. light:', s1.lighting(i.ray.calculate(i.t), l, origin))
#print('Normal Color:', i.pShape.material.color)
#print('Final Color:', s1.lighting(i.ray.calculate(i.t), l, origin) + i.pShape.material.color)
print('')

# eye opposite surface
origin = np.array([0.,0, -1])
l = Light(np.array([0,10,-10]), np.array([1,1,1]))
r = Ray(origin, normalize(s1.center - origin))
i = Intersection(r, RAY_T_MAX)

print('Eye Opposite reflection vector:', s1.intersect(i))
print('Add. light:', s1.lighting(i.ray.calculate(i.t), l, origin))
#print('Normal Color:', i.pShape.material.color)
#print('Final Color:', s1.lighting(i.ray.calculate(i.t), l, origin) + i.pShape.material.color)
print('')

# eye opposite surface
origin = np.array([0.,0, -1])
l = Light(np.array([0,0,10]), np.array([1,1,1]))
r = Ray(origin, normalize(s1.center - origin))
i = Intersection(r, RAY_T_MAX)

print('Behind surface:', s1.intersect(i))
print('Add. light:', s1.lighting(i.ray.calculate(i.t), l, origin))
#print('Normal Color:', i.pShape.material.color)
#print('Final Color:', s1.lighting(i.ray.calculate(i.t), l, origin) + i.pShape.material.color)
print('')

'''

if __name__ == '__main__':

    print('Start Raytracer')

    m = Material(color=np.array([.172,.709,.529]), ambient=0.1, diffuse=0.9, specular=0.9, shinyness=200.)
    # Standard Vals white, 

    s1 = Sphere(np.array([200.,0.,0]), 10, m)
    p = Plane(np.array([0., 0., -10]), np.array([0., 0., 1.]), m)
    s = ShapeSet()
    s.addShape(s1)
    s.addShape(p)

    l = Light(np.array([200, 0, 30]), .5)

    res = 1080
    c = Camera(
        res,                             
        np.array([-100.,0.,100.]),        # origin
        np.array([.8, 0., -.2]),         # direction
        np.array([0., 0., 1.]),         # up
        np.array([0., 1., 0.]),         # right
        15.,                            # fov
        1920 / 1080)                    # ratio


    img = Image()
    RayTrace(img, c, s, l)

    img_rgb = img.get_image()
    cv2_imshow('img',img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
Test
camera = 0,0,-1
normal = 0,0,-1
light = 0,0,-10, color white

'''
