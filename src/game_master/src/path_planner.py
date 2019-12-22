#!/usr/bin/env python
"""
Path Planner Class for Lab 7
Author: Valmik Prabhu
"""

import sys
import rospy
import moveit_commander
from moveit_msgs.msg import OrientationConstraint, Constraints, CollisionObject, Grasp
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive

#TABLE_HEIGHT = -.193 #tall table
TABLE_HEIGHT = -.325 #short table


class PathPlanner(object):
    """
    Path Planning Functionality for Baxter/Sawyer

    We make this a class rather than a script because it bundles up 
    all the code relating to planning in a nice way thus, we can
    easily use the code in different places. This is a staple of
    good object-oriented programming

    Fields:
    _robot: moveit_commander.RobotCommander; for interfacing with the robot
    _scene: moveit_commander.PlanningSceneInterface; the planning scene stores a representation of the environment
    _group: moveit_commander.MoveGroupCommander; the move group is moveit's primary planning class
    _planning_scene_publisher: ros publisher; publishes to the planning scene


    """
    __instance = None
    @staticmethod
    def getInstance():
        """Static access method """
        if PathPlanner.__instance == None:
            PathPlanner()
        return PathPlanner.__instance

    def __init__(self):
        """
        Constructor.

        Inputs:
        group_name: the name of the move_group.
            For Baxter, this would be 'left_arm' or 'right_arm'
            For Sawyer, this would be 'right_arm'
        """
        if PathPlanner.__instance != None:
            raise Exception("this class is a singleton!")
        else:
            PathPlanner.__instance = self

        self.obstacles = {}

        # If the node is shutdown, call this function
        rospy.on_shutdown(self.shutdown)

        # Initialize moveit_commander
        moveit_commander.roscpp_initialize(sys.argv)

        # Initialize the robot
        self._robot = moveit_commander.RobotCommander()

        # Initialize the planning scene
        self._scene = moveit_commander.PlanningSceneInterface()

        # This publishes updates to the planning scene
        self._planning_scene_publisher = rospy.Publisher('/collision_object', CollisionObject, queue_size=10)

        """ Left hand move group"""
        # Instantiate a move group
        self.l_group = moveit_commander.MoveGroupCommander("left_arm")
        self.l_group.set_goal_orientation_tolerance(.01)
        self.l_group.set_goal_position_tolerance(.005)
        ## SET END EFFECTOR ##
        #self._group.set_end_effector_link("left_hand_camera")
        # Set the maximum time MoveIt will try to plan before giving up
        self.l_group.set_planning_time(5)
        # Set the bounds of the workspace
        self.l_group.set_workspace([-2, -2, -2, 2, 2, 2])


        """ Right hand move group"""
        # Instantiate a move group
        self.r_group = moveit_commander.MoveGroupCommander("right_arm")
        self.r_group.set_goal_orientation_tolerance(.01)
        self.r_group.set_goal_position_tolerance(.005)
        self.r_group.set_end_effector_link("right_gripper")
        # Set the maximum time MoveIt will try to plan before giving up
        self.r_group.set_planning_time(5)
        # Set the bounds of the workspace
        self.r_group.set_workspace([-2, -2, -2, 2, 2, 2])


        """dictionary for switching arms"""
        self.arm = {"right": self.r_group, "left": self.l_group}


        # Sleep for a bit to ensure that all inititialization has finished
        rospy.sleep(0.5)

    def shutdown(self):
        """
        Code to run on shutdown. This is good practice for safety

        Currently deletes the object's MoveGroup, so that further commands will do nothing
        """
        self._group = None
        rospy.loginfo("Stopping Path Planner")

    def plan_to_pose(self, target, orientation_constraints, side="left"):
        """
        Generates a plan given an end effector pose subject to orientation constraints

        Inputs:
        target: A geometry_msgs/PoseStamped message containing the end effector pose goal
        orientation_constraints: A list of moveit_msgs/OrientationConstraint messages

        Outputs:
        path: A moveit_msgs/RobotTrajectory path
        """

        self.arm[side].set_pose_target(target)
        self.arm[side].set_start_state_to_current_state()

        #constraints = Constraints()
        #constraints.orientation_constraints = orientation_constraints
        #self._group.set_path_constraints(constraints)

        plan = self.arm[side].plan()

        return plan

    def execute_plan(self, plan, side="left"):
        """
        Uses the robot's built-in controllers to execute a plan

        Inputs:
        plan: a moveit_msgs/RobotTrajectory plan
        """

        return self.arm[side].execute(plan, True)

    def get_pose(self, side="left"):
        return self.arm[side].get_current_pose()

    #main exposed helper function
    def execute_pose(self, target, side="left"):
        """
        Move's to the end effector pose

        Inputs:
        target: A geometry_msgs/PoseStamped message containing the end effector pose goal
        """
        print("executing pose side: ", side)
        plan = self.plan_to_pose(target, None, side=side)
        self.execute_plan(plan, side)
        self.arm[side].clear_pose_targets()

    #secondary exposed helper functions
    def execute_cam_pose(self, target, side="left"):

        self.arm[side].set_end_effector_link(side+"_hand_camera")
        self.arm[side].set_goal_orientation_tolerance(.02)
        self.arm[side].set_goal_position_tolerance(.02)
        self.execute_pose(target, side)
        self.arm[side].set_goal_orientation_tolerance(.01)
        self.arm[side].set_goal_position_tolerance(.005)
        self.arm[side].set_end_effector_link(side+"_gripper")

    def execute_slot_pose(self, target, side="right"):

        self.arm[side].set_end_effector_link(side+"_gripper")
        self.execute_pose(target, side)

    def grab_piece(self):
        #http://docs.ros.org/melodic/api/moveit_msgs/html/msg/Grasp.html

        """
        g = Grasp()
        g.id = "grab_piece"
        g.grasp_pose = pose

        g.pre_grasp_approach.direction.header.frame_id = "base_footprint"
        g.pre_grasp_approach.direction.vector.x = 0.0
        g.pre_grasp_approach.direction.vector.y = 0.0
        g.pre_grasp_approach.direction.vector.z = 0.0
        g.pre_grasp_approach.min_distance = 0.001
        g.pre_grasp_approach.desired_distance = 0.1


        
        joints = self._robot.get_joint_names("right_arm") #arm with gripper
        joint_state = [self._robot.get_joint]
        """
        




    def get_current_pose(self, side="left"):
        return self.arm[side].get_current_pose()

    def add_board(self, pose):
        size = [.22, .22, .02]
        board = PoseStamped()
        name = "board"
        board.header.frame_id = "base"
        board.pose.position = pose.translation
        board.pose.orientation = pose.rotation

        self.remove_obstacle("board")

        self.add_box_obstacle(size, name, board)

        print("Added board collision group")

    def add_table(self, zpos):
        size = [4,4,.01]
        box = PoseStamped()
        name = "table"
        box.header.frame_id = "base"

        #x,y,z position
        box.pose.position.x = 0
        box.pose.position.y = 0
        box.pose.position.z = zpos

        #orientation as quaternion
        box.pose.orientation.x = 0.0
        box.pose.orientation.y = 0.0
        box.pose.orientation.z = 0.0
        box.pose.orientation.w = 1.0

        self.remove_obstacle("table")

        self.add_box_obstacle(size, name, box)

        print("Added table collision group")

    def add_box_obstacle(self, size, name, pose):
        """
        Adds a rectangular prism obstacle to the planning scene

        Inputs:
        size: 3x' ndarray; (x, y, z) size of the box (in the box's body frame)
        name: unique name of the obstacle (used for adding and removing)
        pose: geometry_msgs/PoseStamped object for the CoM of the box in relation to some frame
        """    

        # Create a CollisionObject, which will be added to the planning scene
        co = CollisionObject()
        co.operation = CollisionObject.ADD
        co.id = name
        co.header = pose.header

        # Create a box primitive, which will be inside the CollisionObject
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = size

        # Fill the collision object with primitive(s)
        co.primitives = [box]
        co.primitive_poses = [pose.pose]

        self.obstacles[name] = co

        # Publish the object
        self._planning_scene_publisher.publish(co)

    def stash_table(self):
        self.remove_obstacle("table")

    def unstash_table(self):
        self._planning_scene_publisher.publish(self.obstacles["table"])

    def stash_board(self):
        self.remove_obstacle("board")

    def unstash_board(self):
        self._planning_scene_publisher.publish(self.obstacles["board"])

    def remove_obstacle(self, name):
        """
        Removes an obstacle from the planning scene

        Inputs:
        name: unique name of the obstacle
        """

        co = CollisionObject()
        co.operation = CollisionObject.REMOVE
        co.id = name

        self._planning_scene_publisher.publish(co)
