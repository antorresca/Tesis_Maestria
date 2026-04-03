#!/usr/bin/env python
"""
init_goal_publisher.py
----------------------
Nodo ROS que publica el goal "mantente en la pose actual" al WBC
durante los primeros segundos tras el arranque, evitando que el
controlador persiga un goal por defecto y cause auto-colisión.

Lógica:
  1. Espera el primer mensaje en /mobile_manipulator/data.
  2. Extrae la pose actual del EE (posición + orientación).
  3. Publica esa pose en /mobile_manipulator/desired_traj a 50 Hz
     durante PUBLISH_DURATION segundos.
  4. Se cierra solo.
"""

import math
import rospy
from mobile_manipulator_msgs.msg import Trajectory, Joints
from mobile_manipulator_msgs.msg import MobileManipulator
from geometry_msgs.msg import Transform, Twist, Accel, Vector3, Quaternion

PUBLISH_RATE     = 50    # Hz
PUBLISH_DURATION = 5.0   # segundos publicando el hold goal


def rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll  / 2.0), math.sin(roll  / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw   / 2.0), math.sin(yaw   / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,  # qx
        cr * sp * cy + sr * cp * sy,  # qy
        cr * cp * sy - sr * sp * cy,  # qz
        cr * cp * cy + sr * sp * sy,  # qw
    )


def main():
    rospy.init_node('init_goal_publisher', anonymous=False)

    pub = rospy.Publisher(
        '/mobile_manipulator/desired_traj',
        Trajectory,
        queue_size=1,
    )

    rospy.loginfo("[init_goal_publisher] Esperando estado del robot...")
    data_msg = rospy.wait_for_message(
        '/mobile_manipulator/data',
        MobileManipulator,
        timeout=30.0,
    )

    pos    = data_msg.position.current_position
    orient = data_msg.orientation.current_orient

    qx, qy, qz, qw = rpy_to_quat(orient.roll, orient.pitch, orient.yaw)

    msg = Trajectory()
    msg.pose = Transform(
        translation=Vector3(x=pos.x, y=pos.y, z=pos.z),
        rotation=Quaternion(x=qx, y=qy, z=qz, w=qw),
    )
    msg.vel   = Twist()
    msg.accel = Accel()
    msg.joints = Joints()   # todos en 0.0 (valor por defecto float32)

    rate      = rospy.Rate(PUBLISH_RATE)
    deadline  = rospy.Time.now() + rospy.Duration(PUBLISH_DURATION)

    rospy.loginfo(
        "[init_goal_publisher] Hold goal: pos=(%.3f, %.3f, %.3f) — publicando %.1f s",
        pos.x, pos.y, pos.z, PUBLISH_DURATION,
    )

    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        pub.publish(msg)
        rate.sleep()

    rospy.loginfo("[init_goal_publisher] Listo — WBC estabilizado en pose home.")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
