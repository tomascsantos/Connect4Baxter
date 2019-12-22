#/usr/bin/env python


import os

import cv2
from cv_bridge import CvBridge, CvBridgeError

import argparse
import numpy as np
import matplotlib.pyplot as plt

import rospy


this_file = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = '/'.join(this_file.split('/')[:-2]) + '/img'


def main():
    img = cv2.imread('test.jpg')

    print_img("test", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_img(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name,1080, 720)
    cv2.imshow(name, img)


if __name__ == '__main__':
    main()

