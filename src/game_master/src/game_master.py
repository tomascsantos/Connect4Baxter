#!/usr/bin/env python
import rospy
import time
from game_logic import find_column, update_game_state
from copy import deepcopy
from geometry_msgs.msg import PoseStamped
from vision_master import VisionMaster
from move_master import MoveMaster
from cv_bridge import CvBridge
import ros_numpy
import message_filters
from robot_state import RobotState
from sensor_msgs.msg import Image, CameraInfo
from collections import deque
from game_state import GameState


class GameMaster():

    image_queue = deque([], 5)
    game_state = None

    def __init__(self):

        """
        Initializign Move master will case robot to move to
        the table viewing location
        """

        self.board_prev = None
        self.board_curr = [[0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0],
                           [0,0,0,0,1,2,0],
                           [2,1,2,1,1,2,0]]

        self.poseL = PoseStamped()
        self.poseL.pose.position.x = .174
        self.poseL.pose.position.y = .452
        self.poseL.pose.position.z = -.315

        self.poseR = PoseStamped()
        self.poseR.pose.position.x = .736
        self.poseR.pose.position.y = -.480
        self.poseR.pose.position.z = -0.344

        """Initialize Image Queue"""
        self.bridge = CvBridge()
        self.l_image_sub = message_filters.Subscriber("/cameras/left_hand_camera/image", Image)
        self.l_cam_info_sub = message_filters.Subscriber("/cameras/left_hand_camera/camera_info",
                                  CameraInfo)
        ts = message_filters.ApproximateTimeSynchronizer([self.l_image_sub, 
                                                          self.l_cam_info_sub],
                                                         10, 0.1,
                                                         allow_headerless=True)
        ts.registerCallback(self.callback)


        """init move master stuff"""
        self.mover = MoveMaster(self.poseR.pose.position.z)
        self.viewer = VisionMaster()


        raw_input("press enter to start")
        self.mover.init_grip()
        self.mover.addTable()

        """
        Main Loop

        """
        self.safely_get_state() #keep trying to get state until pressing enter

        while(self.game_state != GameState.GAME_OVER):

            if (self.game_state == GameState.ROBOT_TURN):
                self.safely_pick_and_place_piece()

            elif (self.game_state == GameState.HUMAN_TURN):
                rospy.sleep(3)

            elif (self.game_state == GameState.GAME_OVER_HUMAN):
                #be dissapointed. Rage quit?
                print("RAGE QUIT")
                self.game_state = GameState.GAME_OVER

            elif (self.game_state == GameState.GAME_OVER_ROBOT):
                self.mover.celebrate()
                self.game_state = GameState.GAME_OVER

            else:
                print("must be invalid state")
                rospy.sleep(3)

            self.safely_get_state() #no matter what we want to get the state

        """
            #init vision master stuff
            robot_state = self.mover.go_to_view_board_state()
            rospy.sleep(3) #give it a second for images
            self.board_prev = self.board_curr
            self.board_curr = self.viewer.get_state(robot_state, self.image_queue.pop()[0])
            self.game_state = self.get_game_state(self.board_curr)
            print("game state: ", self.game_state)

            print("play")
            #prep params
            self.mover.go_to_view_board_loc()
            view = self.mover.go_to_table()
            view = RobotState.TOP_VIEW
            rospy.sleep(3) #give it a second to get images
            img, info = self.image_queue.pop()
            #get piece location
            xyz = self.viewer.get_piece(view, img, info, poseL.pose, poseR.pose)
            print(xyz)
            print(poseL)
            print(poseR)
            self.mover.go_to_xyz(xyz)

            raw_input("press enter once piece positioned")
            self.mover.grab()

            col = find_column(self.board_prev, self.board_curr)
            print("col is: ", col)
            self.mover.go_to_slot(col)
            raw_input("press enter to drop")
            self.mover.release()


            self.mover.go_to_view_board_loc()
        """


        rospy.spin()

    def safely_pick_and_place_piece(self):
        self.safely_pick_and_place_piece_helper()
        while (raw_input("place piece successfully? ") != ""):
            self.safely_pick_and_place_piece_helper()

    def safely_pick_and_place_piece_helper(self):
        try:
            self.mover.go_to_table()
            ## update board to obstacles ##
            self.mover.addBoard()
            rospy.sleep(1)

            #self.mover.go_to_view_board_loc()

            rospy.sleep(1)

            img, info = self.image_queue.pop()
            xyz = self.viewer.get_piece(img, info, self.poseL.pose, self.poseR.pose)

            self.mover.go_to_xyz(xyz)
            rospy.sleep(1)

            self.mover.grab()

            col = 6 #find_column(self.board_prev, self.board_curr)

            print("col is: ", col)
            self.mover.go_to_slot(col)
            rospy.sleep(3)
            print("now about to release")
            self.mover.release()
        except Exception as e:
            #print(e)
            raise e
            pass


    def safely_get_state(self):
        self.board_prev = self.board_curr
        self.safely_get_state_helper()
        while(raw_input("get state successful?") != ""):
            self.mover.go_to_table() #reset to the table position
            self.safely_get_state_helper()


    def safely_get_state_helper(self):
        try:
            self.mover.addBoard()
            #self.mover.go_to_view_board_loc()
            self.mover.go_to_view_board_state()
            rospy.sleep(2) #wait for images
            self.board_curr = self.viewer.get_state(self.image_queue.pop()[0])
            self.game_state = update_game_state(self.board_prev, self.board_curr)
            print("game state is:", self.game_state)
        except Exception as e:
            #print(e)
            raise e
            pass


    def callback(self, img, info):
        try:
            rgb_image = ros_numpy.numpify(img)
            #we're doing append and pop() so FIFO
            self.image_queue.append((rgb_image, info))
        except Exception as e:
            rospy.logerr(e)
            return


if __name__ == '__main__':
    rospy.init_node('game_master')
    master = GameMaster()
