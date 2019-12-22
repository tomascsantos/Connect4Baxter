#!/usr/bin/env python

import cv2
from collections import deque
import rospy
import numpy as np
import ros_numpy
from sensor_msgs.msg import Image, CameraInfo
from vision_processing import *
import message_filters
from robot_state import RobotState
from cv_bridge import CvBridge


class VisionMaster():

    __instance = None


    @staticmethod
    def getInstance():
        """Static Access Method"""
        if VisionMaster.__instance == None:
            VisionMaster()
        return VisionMaster.__instance

    def __init__(self):
        print("vision initialized")

    def get_state(self, img):
        rospy.loginfo(type(img))
        return get_board_state(img)

    def get_piece(self, img, info, poseL, poseR):
        xL = poseL.position.x
        yL = poseL.position.y
        zL = poseL.position.z

        xR = poseR.position.x
        yR = poseR.position.y
        zR = poseR.position.z


        pieces = find_the_pieces(img)

        #grab middle piece
        l = len(pieces)
        i = int(l/2)
        piece = pieces[i]
        da_piece = []
        da_piece.append(piece)

        markup_img_at_points(img, da_piece, "corners")

        piece_u = piece[0]
        piece_v = piece[1]

        u_and_v = find_the_green_pieces(img) #top right then bottom left

        markup_img_at_points(img, u_and_v, "corners")

        print("u and v: ", u_and_v)
        print()

        u_l = u_and_v[1][0]
        v_l = u_and_v[1][1]

        u_r = u_and_v[0][0]
        v_r = u_and_v[0][1]

        #find x:
        delta_vd = float(piece_v - v_r)
        delta_v = float(v_l - v_r)
        delta_x = float(xR - xL)
        delta_xd = delta_vd / delta_v * delta_x
        x = xR - delta_xd

        print("delta_vd: ", delta_vd, "delta_v", delta_v, "delta_x", delta_x, "delta_xd", delta_xd, "x: ", x)

        #find y
        delta_ud = float(u_r - piece_u)
        delta_u = float(u_r - u_l)
        delta_y = float(yL - yR)
        delta_yd = delta_ud / delta_u * delta_y
        y = yR + delta_yd

        print("delta_ud: ", delta_ud, "delta_u", delta_u, "delta_y", delta_y, "delta_yd", delta_yd, "y: ", y)


        return [x,y,zR+.04]

def get_camera_matrix(camera_info_msg):
    return np.reshape(camera_info_msg.K, (3,3))


if __name__ == '__main__':
    vision = VisionMaster.getInstance()

