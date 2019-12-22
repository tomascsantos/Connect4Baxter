#!/usr/bin/env python
import rospy
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
from baxter_interface import gripper as robot_gripper
import numpy as np
from numpy import linalg
import sys

def main(robo):
    #Wait for the IK service to become available
    rospy.wait_for_service('compute_ik')
    rospy.init_node('service_query')
    right_gripper = robot_gripper.Gripper('right')
    arm = 'right'
    #Create the function used to call the service
    compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
    if robo == 'sawyer':
    	arm = 'right'
    count = 0
    print('Calibrating...')
    right_gripper.calibrate()
    rospy.sleep(2.0)
    while not rospy.is_shutdown():
        raw_input('Press [ Enter ]: ')
        
        #Construct the request
        request = GetPositionIKRequest()
        request.ik_request.group_name = arm + "_arm"

        #Alan does not have a gripper so replace link with 'right_wrist' instead
        link = arm + "_gripper"
        if robo == 'sawyer':
        	link += '_tip'

        request.ik_request.ik_link_name = link
        request.ik_request.attempts = 20
        request.ik_request.pose_stamped.header.frame_id = "base"
        
	x = [.669, 0, .656, .664, 0]
	y = [-.609, 0, -.480, -.332, 0]
	z = [-.050, 0, .097, -.050, 0]
	
	if (count == 0 or count == 2 or count == 3):	

		#Set the desired orientation for the end effector HERE
		request.ik_request.pose_stamped.pose.position.x = x[count]
		request.ik_request.pose_stamped.pose.position.y = y[count]
		request.ik_request.pose_stamped.pose.position.z = z[count]
		request.ik_request.pose_stamped.pose.orientation.x = 0.0
		request.ik_request.pose_stamped.pose.orientation.y = 1.0
		request.ik_request.pose_stamped.pose.orientation.z = 0.0
		request.ik_request.pose_stamped.pose.orientation.w = 0.0
		
		try:
		    #Send the request to the service
		    response = compute_ik(request)
		    
		    #Print the response HERE
		    print(response)
		    group = MoveGroupCommander(arm + "_arm")

		    # Setting position and orientation target
		    group.set_pose_target(request.ik_request.pose_stamped)

		    # TRY THIS
		    # Setting just the position without specifying the orientation
		    #group.set_posIition_target([0.5, 0.5, 0.0])

		    # Plan IK and execute
		    group.go()
		    
		except rospy.ServiceException, e:
		    print "Service call failed: %s"%e


	#Close the right gripper
	if count == 1:
		print('Closing...')
		right_gripper.close()
		rospy.sleep(1.0)
	elif count == 4:
		#Open the right gripper
		print('Opening...')
		right_gripper.open()
		rospy.sleep(1.0)
	count+=1


#Python's syntax for a main() method
if __name__ == '__main__':
    main(sys.argv[1])

