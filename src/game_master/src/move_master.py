#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from ar_track_alvar_msgs.msg import AlvarMarkers
from moveit_commander import MoveGroupCommander
from baxter_interface import gripper as robot_gripper
import numpy as np
import tf2_ros
from numpy import linalg
from path_planner import PathPlanner
from robot_state import RobotState


TABLE_TAG = 3
TABLE_TAG2 = 9
BOARD_TAG = 12

"""
Initializing this class causes the robot to move to the view
table location and add the table as an obstacle. It then
initializes a service.

Service: "game_move_controller"
    input:
        string cmd_type             --Which location to move to or "dynamic"
        geometry_msgs/Pose pose     --If "dynamic" then move to this pose
    output:
        string resp                 --If we need something from vision
"""

class MoveMaster():

    board_pose = None
    table_xyz = None


    def __init__(self, table_height):

        self.table_height = table_height
        self.left_tag = None
        self.right_tag = None

        print("beginning movement master initialization")
        #Initialize the Planner
        self.planner = PathPlanner()
        self.gripper = robot_gripper.Gripper('right')
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        r = rospy.Rate(10)


    def init_grip(self):
        self.gripper.calibrate()
        params = self.gripper.valid_parameters()
        print(params)
        rospy.sleep(2)

    def grab(self):
        #get current spot
        rospy.sleep(1)
        trans = self.getTrans("base", "right_gripper")
        goal = PoseStamped()
        goal.header.frame_id = "base"
        goal.pose.position = trans.transform.translation
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 1.0
        goal.pose.orientation.z = 0.0
        goal.pose.orientation.w = 0.0

        #get rid of table obstancle so move it doesn't fail
        self.planner.stash_table()
        rospy.sleep(1)

        ###move down from hover position
        goal.pose.position.z -= .05 #we offset the hover position by .04
        print("grab move down goal is: ", goal)
        self.planner.execute_pose(goal, side="right")

        ##close gripper
        rospy.sleep(2)
        self.gripper.close()
        rospy.sleep(.1)

        ##move up a bit
        goal.pose.position = trans.transform.translation
        goal.pose.position.z += .3
        print("grab move up goal is: ", goal)
        self.planner.execute_pose(goal, side="right")

        #put the table back
        self.planner.unstash_table()
        rospy.sleep(.5)




    def release(self):
        print("grabing transform")
        trans = self.getTrans("base", "right_gripper")

        goal = PoseStamped()
        goal.header.frame_id = "base"
        goal.pose.position = trans.transform.translation
        goal.pose.orientation = trans.transform.rotation
        rospy.sleep(1)

        #remove board during this
        self.planner.stash_board()

        #go down a bit
        goal.pose.position.z -= .085
        self.planner.execute_pose(goal, side="right")

        ##open gripper
        rospy.sleep(.5)
        self.gripper.open()
        rospy.sleep(.5)

        #go up a bit
        goal.pose.position.z += .06
        self.planner.execute_pose(goal, side="right")

        self.planner.unstash_board()

    def addTable(self):
        print("[move_master] moving to table view")
        self.go_to_table()
        #trans = self.getTransMarker(TABLE_TAG) #Grab the position of the table
        #pos = trans.transform.translation #extract the position from the transformation
        #self.table_xyz = pos
        self.planner.add_table(self.table_height)
        rospy.sleep(.5)


    def addBoard(self):
        """We start the game with right hand looking at the
        board, so we save this state as board pose"""
        self.go_to_view_board_loc()
        rospy.sleep(.5)
        print("adding board")
        trans = self.getTransMarker(BOARD_TAG)
        self.board_pose = trans.transform
        self.planner.add_board(self.board_pose)
        rospy.sleep(1)


    slots = dict({1:-.085, 2:-.06, 3:-.025, 4:.01, 5:.045, 6:.08, 7:.105})
    #offset x dir += .01
    def go_to_slot(self, slot):
        """
        Ar tag is centered on board and board is:
        .26m wide --> board_pose.x + .13 //center, .035m for each slot mid2mid
        .20m tall --> board_pose.z + .10
        .02m fat  --> board_pose.y + .01

        -9.5 -7 -3.5  0  3.5  7  9.5
        |   |   |   |   |   |   |   |
        """
        assert self.board_pose != None, "board position is null"

        board_xyz = self.board_pose.translation
        goal = PoseStamped()
        goal.header.frame_id = "base"
        goal.pose.position.x =board_xyz.x + self.slots[slot]
        goal.pose.position.y =board_xyz.y + .01
        goal.pose.position.z =board_xyz.z + .23

        goal.pose.orientation.x = 0.5
        goal.pose.orientation.y = 0.5
        goal.pose.orientation.z = 0.5
        goal.pose.orientation.w = -0.5

        print("goal was: ", goal)
        print("board is at: ", board_xyz)


        self.planner.execute_slot_pose(goal)        #right hand



    def getTransMarker(self, ar_marker_num, side="left", cam=False):
        """
        Gets the transform of the requested ar tag
        Returns: a geometry_msg/PoseStamped transform
        """
        target = "base"
        if cam:
            target = side + "_hand_camera"
        source = "ar_marker_" + str(ar_marker_num)

        print("target: ", target, "source",source)
        return self.getTrans(target, source)


    def getTrans(self, target, source):
        trans = None
        while not rospy.is_shutdown() and trans == None:
            try:
                trans = self.tfBuffer.lookup_transform(target,source,rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                print("failed to get ", source,"to",target,"transform")
            rospy.sleep(.5)
        return trans




    """For the master to call"""
    def get_board(self):
        return self.board_pose

    def get_table_markers(self):
        trans1 = self.getTransMarker(TABLE_TAG, cam=True).transform
        trans2 = self.getTransMarker(TABLE_TAG2, cam=True).transform
        return (trans1, trans2)

    def go_to_xyz(self, xyz):
        pose = PoseStamped()
        pose.header.frame_id = "base"

        pose.pose.position.x = xyz[0]
        pose.pose.position.y = xyz[1]
        pose.pose.position.z = xyz[2]

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 1.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0
        self.planner.execute_pose(pose, side="right")     #right hand


    def go_to_table(self):
        print("going to table")
        pose = PoseStamped()
        pose.header.frame_id = "base"

        pose.pose.position.x = 0.6
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.2

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 1.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0
        self.planner.execute_pose(pose)     #left hand
        return RobotState.TOP_VIEW

    def go_to_view_board_loc(self):
        pose = PoseStamped()
        pose.header.frame_id = "base"


        pose.pose.position.x = .444
        pose.pose.position.y = -.457
        pose.pose.position.z = .080
        pose.pose.orientation.x = 0.9004494
        pose.pose.orientation.y = 0.0130065
        pose.pose.orientation.z = 0.0470235
        pose.pose.orientation.w = -0.4322157

        self.planner.execute_cam_pose(pose, side="right")
        return RobotState.BOARD_VIEW


    def go_to_view_board_state(self):
        pose = PoseStamped()
        pose.header.frame_id = "base"
        pose.pose.position.x = self.board_pose.translation.x
        pose.pose.position.y = self.board_pose.translation.y + .12 #.11 -> .15
        pose.pose.position.z = self.board_pose.translation.z + .01
        pose.pose.orientation = self.board_pose.rotation

        self.planner.execute_cam_pose(pose)     #left hand
        return RobotState.BOARD_VIEW

if __name__ == '__main__':
    mover = MoveMaster.getInstance()
