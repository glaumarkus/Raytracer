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

                if shape.intersect(i, light):
                    '''
                    local_color = i.pShape.material.color
                    light_color = black
                    if isinstance(i.pShape, Sphere):
                        light_color = i.pShape.lighting(i.ray.calculate(i.t), light, camera.origin)
                        print(local_color, light_color)
                    '''
                    #if isinstance(i.pShape, Sphere):
                    #    local_color = 
                    #print(i.pShape.lighting(i.ray.calculate(i.t), light, -r.direction)) #* i.pShape.material.color
                    #else:
                    #    local_color = i.color
                    image.image[x,y] = clamp_color(i.color)
                else:
                    image.image[x,y] = black



if __name__ == '__main__':

    print('Start Raytracer')

    m = Material(color=np.array([.172,.709,.529]), ambient=0.1, diffuse=0.9, specular=0.9, shinyness=80.)
    m1 = Material(color=np.array([.172,.709,.529]), ambient=0.2, diffuse=.9, specular=0.9, shinyness=200.)
    m2 = Material(color=np.array([.172,.709,.529]), ambient=0.2, diffuse=.9, specular=0.9, shinyness=200.)
    m3 = Material(color=np.array([.172,.709,.529]), ambient=0.2, diffuse=.9, specular=0.9, shinyness=200.)
    m4 = Material(color=np.array([.72,.709,.529]), ambient=0.4, diffuse=0.9, specular=0.9, shinyness=200.)
    m5 = Material(color=np.array([.72,.709,.529]), ambient=0.5, diffuse=0.9, specular=0.9, shinyness=200.)
    # Standard Vals white, 

    s = ShapeSet()
    s1 = Sphere(np.array([200.,0.,0]), 15, m1, s)
    s2 = Sphere(np.array([200.,45.,0]), 15, m2, s)
    s3 = Sphere(np.array([200.,-45.,0]), 15, m3, s)

    p = Plane(np.array([0., 0., -30]), np.array([0., 0., 1.]), m, s)
    #s = ShapeSet()
    s.addShape(s1)
    s.addShape(s2)
    s.addShape(s3)
    s.addShape(p)

    l = Light(np.array([50, 100, 200]), 1.)

    res = 800
    c = Camera(
        res,                             
        np.array([-200.,0.,100.]),        # origin
        np.array([.8, 0., -.2]),         # direction
        np.array([0., 0., 1.]),         # up
        np.array([0., 1., 0.]),         # right
        15.,                            # fov
        1920 / 1080)                    # ratio


    img = Image()
    RayTrace(img, c, s, l)

    img_rgb = img.get_image()
    cv2.imwrite('test.png', img_rgb) 
    #cv2_imshow('img',img_rgb)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


