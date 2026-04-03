cmake_minimum_required(VERSION 3.0.2)
project(mobile_manipulator_unal_description)

find_package(catkin REQUIRED COMPONENTS
  controller_manager
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  mobile_manipulator_msgs
)

################################################
## Declare ROS messages, services and actions ##
################################################

###################################
## catkin specific configuration ##
###################################

catkin_package(
  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs mobile_manipulator_msgs
)

###########
## Build ##
###########

catkin_install_python(PROGRAMS
  scripts/init_goal_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)