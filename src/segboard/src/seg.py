#!/usr/bin/env python


import os

import cv2
from cv_bridge import CvBridge, CvBridgeError

import argparse
import numpy as np
import matplotlib.pyplot as plt

import rospy
import intera_interface


this_file = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = '/'.join(this_file.split('/')[:-2]) + '/img'
#TODO
#Find corners of board and zoom in (find largest cluster? - will need all four corners)
#if clusters (of pieces seg) arent greater than threshold, make black [this is on zoomed in seg pieces]
#divide new zoomed in array into 7 columns and 6 rows (aka if new array is 100 by 100), we have a step of 14 horiztonally and 17 vertically.
#if a one appears in any of these steps, we add the corresponding piece to our CUSTUM data array (6 by 7) 
#profit 

def read_image(img_name, grayscale=False):
    """ reads an image

    Parameters
    ----------
    img_name : str
        name of image
    grayscale : boolean
        true if image is in grayscale, false o/w
    Returns
    -------
    ndarray
        an array representing the image read (w/ extension)
    """

    if not grayscale:
        img = cv2.imread(img_name)
    else:
        img = cv2.imread(img_name, 0)

    return img

def print_img(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name,1080, 720)
    cv2.imshow(name, img)



def threshold(img_name, color):
    img = cv2.imread(img_name)
    print('img', img.shape)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print('img_hsv', img_hsv.shape)
    lower_r = np.array([0,80,80])
    upper_r = np.array([10,255,255])

    lower_b = np.array([110,80,80])
    upper_b = np.array([130,255,255])

    lower_y = np.array([20,80,80])
    upper_y = np.array([30,255,255])

    red = cv2.inRange(img_hsv,lower_r,upper_r)
    blue = cv2.inRange(img_hsv,lower_b,upper_b)
    yellow = cv2.inRange(img_hsv,lower_y,upper_y)
    print('blue', blue.shape)
    if (color == 'RED'):
        # print_img('red',red)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return red
    if (color == 'BLUE'):
        print_img('blue', blue)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return blue
    if (color == 'YELLOW'):
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print_img('yellow',yellow)
        return yellow
    return img_hsv

def test_thresh(img, color):
    thresh = threshold(img, color)
    show_image(thresh, title='thresh naive')
    cv2.imwrite(IMG_DIR + "/thresh.jpg", thresh.astype('uint8') * 255)


def show_image(img_name, title='Fig', grayscale=False):
    """show the  as a matplotlib figure
    Parameters
    ----------
    img_name : str
        name of image
    tile : str
        title to give the figure shown
    grayscale : boolean
        true if image is in grayscale, false o/w
    """

    if not grayscale:
        plt.imshow(cv2.cvtColor(img_name, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.show()
    else:
        plt.imshow(img_name, cmap='gray')
        plt.title(title)
        plt.show()



def show_image_callback(img_data, window_name):
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(img_data, "bgr8")
    except CvBridgeError, err:
        rospy.logerr(err)
        return

    """ Segment the image """
    #define color bounds RED (hue, saturation, value)
    lower_r = np.array([0,80,80])
    upper_r = np.array([10,255,255])
    #define color bounds YELLOW (hue, saturation, value)
    lower_y = np.array([20,80,80])
    upper_y = np.array([30, 255, 255])
    #define color bounds BLUE (hue, saturation, value)
    lower_b = np.array([110,80,20])
    upper_b = np.array([130, 255, 125])

    #represent it in hsv colorspace
    img_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    #use the thresholds to create different segment maps
    blue = cv2.inRange(img_hsv, lower_b, upper_b)
    #yellow = cv2.inRange(img_hsv, lower_y, upper_y)



    cv_win_name = cv2.namedWindow(window_name, 0)
<<<<<<< HEAD
    #cv2.setMouseCallback(window_name, printHSV)

#remove this eventually
    #while(1):
    cv2.imshow(cv_win_name, blue)
    cv2.waitKey(3)
    cv2.imshow(cv_win_name, cv_image)
    cv2.waitKey(3)

#    cv2.setMouseCallback(window_name, printHSV)

#remove this eventually
#    while(1):
#        cv2.imshow(cv_win_name, img_hsv)
#
    #refresh the image on the screen

def printHSV(event, x, y, flags, param):
    print("we did it!")
    if event:
        print(event)
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("buttonclick")
        h = hsv.val[0]
        s = hsv.val[1]
        v = hsv.val[2]
        print("hsv: ", h, " : ", s, " :  ", v, " : ")


def setupCamera():
    """ Make sure the camera's exist """
    rp = intera_interface.RobotParams()
    valid_cameras = rp.get_camera_names()
    if not valid_cameras:
        rp.log_message(("Cannot detect any camera_config"
        " parameters on this robot. Exiting."), "ERROR")
        return

    """ Setup the node """
    rospy.init_node('camera_display', anonymous=True)
    cameras = intera_interface.Cameras()
    #camera = "right_hand_camera"
    camera = "head_camera"

    window_name = "Right Hand Camera Window"
    if not cameras.verify_camera_exists(camera):
        rospy.logerr("Could not detect the specified camera, exiting the example.")
        return
    rospy.loginfo("Opening camera '{0}'...".format(camera))
    cameras.start_streaming(camera)

    """ Call the callback """
    cameras.set_callback(camera, show_image_callback, callback_args=window_name)

    """ Configure the camera  """
    success = cameras.set_gain(camera, -1) #-1 for autogain
    if not success:
        rospy.logger("failed to set gain")
        return
    success = cameras.set_exposure(camera, -1)
    if not success:
        rospy.logger("failed to set exposure")
        return

    def clean_shutdown():
        print("Shutting down camera_display node.")
        cv2.destroyAllWindows()

    rospy.on_shutdown(clean_shutdown)
    rospy.loginfo("Camera_display node running. Ctrl-c to quit")
    rospy.spin()

#___Disjoint Set____#


if __name__ == '__main__':
    #blue = threshold(IMG_DIR + '/IMG-1115.jpg', 'BLUE')
    #print(blue)
    #print(blue.shape)


    #x = np.where(np.any(mask == 0, axis=-1))
    # non_black = np.any(blue != [0,0,0], axis=-1)

    # print_img('test',blue[non_black])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    hsv = threshold(IMG_DIR + '/IMG-1115.jpg', 'BLUE')
    print(np.any(hsv[:]))
    print(hsv)
    #Disjoint
    flattened = hsv.flatten()
