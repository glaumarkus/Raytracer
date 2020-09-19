import math
import numpy as np
import cv2
from cv2 import imshow as cv2_imshow

from ray import Ray
from intersection import Intersection
from shapes import Material, ShapeSet, Plane, Sphere, Light, CheckBoard
from observer import Image, Camera


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

    print('Duration:')
    sep = int(image.H / 40)
    counter = 0
    counter2 = 0

    for x in range(image.H):

        counter += 1
        if counter > sep:
            counter = 0
            counter2 += 1
            print('#', end='', flush=True)

        for y in range(image.W):
            if camera.getRay(x,y):
                r = camera.getRay(x,y)
                i = Intersection(r, RAY_T_MAX)
                if shape.intersect(i, light):
                    image.image[x,y] = clamp_color(i.color)
                else:
                    image.image[x,y] = black



if __name__ == '__main__':

    print('Start Raytracer')



    # color blueprints
    blue = np.array([0.498, 0.403, 0.])
    gray = np.array([0.48, 0.48, 0.48])
    light_green = np.array([0.756, 0.929, 0.862])
    light_pink = np.array([0.666, 0.756, 1.])
    
    red = np.array([.047, .09, .447])
    orange = np.array([.137, .568, .901])
    green = np.array([.274, .627, .431])
    white = np.array([1., 1., 1])

    MatGray = Material(color=gray, ambient=0.7, diffuse=0.9, specular=1., shinyness=80.)
    MatBlue = Material(color=blue, ambient=0.7, diffuse=0.9, specular=.9, shinyness=80.)
    MatRed = Material(color=red, ambient=0.7, diffuse=0.9, specular=.9, shinyness=80.)
    MatGreen = Material(color=green, ambient=0.7, diffuse=0.6, specular=1., shinyness=80.)
    MatOrange = Material(color=orange, ambient=0.1, diffuse=0.9, specular=1., shinyness=80.)
    MatWhite = Material(color=white, ambient=0.1, diffuse=0.9, specular=1., shinyness=80.)
    MatBack = Material(color=light_green, ambient=0.1, diffuse=0.9, specular=1., shinyness=80.)
    MatBack2 = Material(color=light_pink, ambient=0.1, diffuse=0.9, specular=1., shinyness=80.)

    # build shapes 
    s1 = Sphere(np.array([170.,0.,20]), 20, MatBlue)
    s2 = Sphere(np.array([250.,75.,40]), 40, MatRed)
    s3 = Sphere(np.array([260.,-35.,35]), 35, MatGreen)
    s4 = Sphere(np.array([200.,-15.,45]), 5, MatOrange)

    back = Plane(np.array([300., 0., 0.]), np.array([1., 0., 0.]), MatBack)
    right = Plane(np.array([0., -170., 0.]), np.array([0, -1., 0.]), MatBack2)
    #b = Plane(np.array([0., 0., 0]), np.array([0., 0., 1.]), MatGray)
    b = CheckBoard(np.array([0., 0., 0.]), np.array([0., 0., 1.]), MatGray)

    if True:

        s = ShapeSet()
        s.addShape(s1)
        s.addShape(s2)
        s.addShape(s3)
        s.addShape(s4)

        s.addShape(b)
        s.addShape(back)
        s.addShape(right)

    else:
        s = ShapeSet()
        s.addShape(s1)
        s.addShape(b)


        pass

    l = Light(np.array([-130, 170, 230]), .8)

    res = 100
    c = Camera(
        res,                             
        np.array([-130., 30.,200.]),        # origin
        np.array([0.66, 0, -.33]),         # direction
        np.array([0., 0., 1.]),         # up
        np.array([0., 1., 0.]),         # right
        20.,                            # fov
        1920 / 1080)                    # ratio


    img = Image()
    RayTrace(img, c, s, l)

    img_rgb = img.get_image()
    img_rgb *= 255

    cv2.imwrite('test.jpg', img_rgb.astype(int)) 
    #cv2_imshow('img',img_rgb.astype(int))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


