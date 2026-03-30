"""
robot_interface.py
------------------
Thin bridge between external AI modules (DRL, Task Planning) and the
robot WBC running inside Docker via rosbridge WebSocket.

Responsibilities:
  - send_goal()  : publish a Cartesian target → WBC tracks it
  - get_state()  : return latest robot state (EE pose + joints)
  - reset()      : reset Gazebo world to spawn state, hold WBC in place

NOT responsible for:
  - Any control logic or corrections (that lives in the WBC / config.yaml)
  - Selecting WBC mode (mobile_manipulator / only_manipulator / mobile_robot)
  - Computing rewards or observations for DRL (that lives in the Gym env)
"""

import time
import threading
import roslibpy


class RobotInterface:
    """WebSocket bridge to the robot WBC.

    Usage:
        ri = RobotInterface(host='localhost', port=9090)
        ri.connect()
        ri.wait_for_state()

        ri.send_goal(x=1.0, y=0.0, z=0.5)
        state = ri.get_state()   # dict with ee_pos, ee_rpy, joints

        initial_state = ri.reset()

        ri.disconnect()
    """

    _TOPIC_GOAL  = '/mobile_manipulator/desired_traj'
    _TOPIC_STATE = '/mobile_manipulator/data'
    _SVC_RESET   = '/gazebo/reset_world'

    def __init__(self, host='localhost', port=9090):
        self._host = host
        self._port = port
        self._client    = None
        self._goal_pub  = None
        self._state_sub = None
        self._state     = None
        self._lock      = threading.Lock()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        """Connect to rosbridge and start listening to robot state."""
        self._client = roslibpy.Ros(host=self._host, port=self._port)
        self._client.run()

        self._goal_pub = roslibpy.Topic(
            self._client,
            self._TOPIC_GOAL,
            'mobile_manipulator_msgs/Trajectory',
        )
        self._goal_pub.advertise()

        self._state_sub = roslibpy.Topic(
            self._client,
            self._TOPIC_STATE,
            'mobile_manipulator_msgs/MobileManipulator',
        )
        self._state_sub.subscribe(self._on_state)

    def disconnect(self):
        """Cleanly unsubscribe and close the WebSocket connection."""
        if self._state_sub:
            self._state_sub.unsubscribe()
        if self._goal_pub:
            self._goal_pub.unadvertise()
        if self._client:
            self._client.terminate()

    @property
    def connected(self):
        return self._client is not None and self._client.is_connected

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _on_state(self, msg):
        with self._lock:
            self._state = msg

    def get_state(self):
        """Return the latest robot state as a plain dict, or None if not ready.

        Keys:
            ee_pos  : [x, y, z]       end-effector position (m)
            ee_rpy  : [roll, pitch, yaw]  end-effector orientation (rad)
            joints  : dict with keys mobjoint1..3, joint1..6  (rad)
        """
        with self._lock:
            if self._state is None:
                return None
            s = self._state

        pos    = s['position']['current_position']
        orient = s['orientation']['current_orient']
        joints = s['joints']['actual']

        return {
            'ee_pos': [pos['x'], pos['y'], pos['z']],
            'ee_rpy': [orient['roll'], orient['pitch'], orient['yaw']],
            'joints': dict(joints),
        }

    def wait_for_state(self, timeout=5.0):
        """Block until the first state message arrives.

        Returns True if a state was received, False on timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._state is not None:
                    return True
            time.sleep(0.01)
        return False

    # ------------------------------------------------------------------
    # Goal
    # ------------------------------------------------------------------

    def send_goal(self, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        """Publish a Cartesian goal to the WBC.

        The WBC runs in ConstVel mode and manages velocity/acceleration
        internally, so vel and accel are left at zero here.

        Args:
            x, y, z       : desired end-effector position in world frame (m)
            qx, qy, qz, qw: desired end-effector orientation as quaternion
                            (default: identity — no rotation)
        """
        msg = roslibpy.Message({
            'pose': {
                'translation': {'x': float(x), 'y': float(y), 'z': float(z)},
                'rotation':    {'x': float(qx), 'y': float(qy),
                                'z': float(qz), 'w': float(qw)},
            },
            'vel': {
                'linear':  {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            },
            'accel': {
                'linear':  {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            },
            # Desired joint posture for the WBC null-space task.
            # All zeros = neutral posture reference. Adjust if a specific
            # home configuration is defined for the arm.
            'joints': {
                'mobjoint1': 0.0, 'mobjoint2': 0.0, 'mobjoint3': 0.0,
                'joint1': 0.0, 'joint2': 0.0, 'joint3': 0.0,
                'joint4': 0.0, 'joint5': 0.0, 'joint6': 0.0,
            },
        })
        self._goal_pub.publish(msg)

    # ------------------------------------------------------------------
    # Reset (training episodes)
    # ------------------------------------------------------------------

    def reset(self, settle_time=0.3):
        """Reset the simulation for a new training episode.

        Steps:
          1. /gazebo/reset_world  — resets robot to spawn pose (0,0,0),
                                    no simulation time jump.
          2. Wait settle_time     — lets Gazebo apply the reset and the
                                    state topic to reflect the new pose.
          3. Read post-reset state and send it as the WBC target so the
             controller holds position instead of chasing the previous
             episode's last goal.

        Args:
            settle_time: seconds to wait after reset_world (default 0.3 s)

        Returns:
            Initial state dict (same format as get_state()), or None if
            no state was received within settle_time.
        """
        svc = roslibpy.Service(
            self._client,
            self._SVC_RESET,
            'std_srvs/Empty',
        )
        svc.call(roslibpy.ServiceRequest())

        time.sleep(settle_time)

        state = self.get_state()
        if state is not None:
            self.send_goal(*state['ee_pos'])

        return state
