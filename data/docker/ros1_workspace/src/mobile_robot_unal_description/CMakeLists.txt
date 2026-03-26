cmake_minimum_required(VERSION 3.0.2)
project(mobile_robot_unal_description)

# 1. Buscar paquetes de ROS
find_package(catkin REQUIRED COMPONENTS
  controller_manager
  roscpp
  sensor_msgs
  std_msgs
  nav_msgs
  tf2
  tf2_ros
  gazebo_msgs
  tf
)

# 2. Buscar Eigen3 correctamente usando el módulo de CMake
find_package(Eigen3 REQUIRED)

# 3. Configuración de Catkin (IMPORTANTE: Eigen3 debe estar aquí si otros paquetes lo usarán)
catkin_package(
  INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIRS} # Usar la variable que genera find_package
  CATKIN_DEPENDS 
    controller_manager roscpp sensor_msgs std_msgs nav_msgs tf2 tf2_ros gazebo_msgs tf
  DEPENDS Eigen3
)

# 4. Directorios de inclusión (Antes de los ejecutables)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS} # Esto asegura que el compilador vea los .h de Eigen
)

###########
## Build ##
###########

add_executable(odometry_node
  src/mobile_odom_node.cpp
  src/MobileOdom.cpp
)

target_link_libraries(odometry_node
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES} # Aunque Eigen sea 'header-only', es buena práctica
)

add_executable(world_odom_corrector
  src/world_odom_corrector.cpp
)

target_link_libraries(world_odom_corrector
  ${catkin_LIBRARIES}
)

add_executable(gazebo_to_ekf src/gazebo_to_ekf.cpp)
target_link_libraries(gazebo_to_ekf ${catkin_LIBRARIES})