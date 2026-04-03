"""
mobile_manipulator_env.py — Entorno Gymnasium para el robot manipulador móvil.

Implementa la interfaz gym.Env sobre robot_interface.RobotInterface + sensor_utils.
El DRL elige una acción discreta (Δpose incremental) y recibe:
    - Observación: vector de 9 valores normalizados.
    - Recompensa:  shaping de distancia + colisión + éxito + penalizaciones.

Uso típico (training loop):
    env = MobileManipulatorEnv(host='localhost', port=9090,
                               obstacle_models=['unit_box_1'])
    obs, info = env.reset(options={'pose_goal': pose_goal_dict})
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()

Compatibilidad: gymnasium >= 0.29  (SB3 >= 2.0)
"""

import math
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from ..robot_interface.robot_interface import RobotInterface
    from .sensor_utils import get_obstacle_info
    from .config import (
        ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
        ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_STAY,
        STEP_XY, STEP_Z, TURN_ANGLE_RAD, SLERP_ALPHA,
        MAX_DIST, GOAL_THR, COLLISION_RADIUS, DETECTION_RADIUS, MAX_STEPS,
        STEP_SLEEP, RESET_SETTLE_TIME, RESET_EXTRA_SLEEP,
        REWARD_COLLISION, REWARD_SUCCESS, REWARD_STEP, REWARD_STAY,
        DEFAULT_TRAINING_GOALS, WORKSPACE_X, WORKSPACE_Y, WORKSPACE_Z,
        _EE_HOME,
    )
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from robot_interface.robot_interface import RobotInterface
    from drl.sensor_utils import get_obstacle_info
    from drl.config import (
        ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
        ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_STAY,
        STEP_XY, STEP_Z, TURN_ANGLE_RAD, SLERP_ALPHA,
        MAX_DIST, GOAL_THR, COLLISION_RADIUS, DETECTION_RADIUS, MAX_STEPS,
        STEP_SLEEP, RESET_SETTLE_TIME, RESET_EXTRA_SLEEP,
        REWARD_COLLISION, REWARD_SUCCESS, REWARD_STEP, REWARD_STAY,
        DEFAULT_TRAINING_GOALS, WORKSPACE_X, WORKSPACE_Y, WORKSPACE_Z,
        _EE_HOME,
    )


# ---------------------------------------------------------------------------
# Utilidades de cuaterniones (sin dependencias externas)
# ---------------------------------------------------------------------------

def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler ZYX (convención ROS) → cuaternión [qx, qy, qz, qw]."""
    cr, sr = math.cos(roll  / 2), math.sin(roll  / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw   / 2), math.sin(yaw   / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,   # qx
        cr * sp * cy + sr * cp * sy,   # qy
        cr * cp * sy - sr * sp * cy,   # qz
        cr * cp * cy + sr * sp * sy,   # qw
    ], dtype=np.float32)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolación esférica entre q0 y q1 con fracción alpha ∈ [0,1]."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:       # garantizar camino corto
        q1  = -q1
        dot = -dot
    dot = min(dot, 1.0)
    theta = math.acos(dot)
    if theta < 1e-6:    # cuaterniones casi idénticos
        return q0.copy()
    s = math.sin(theta)
    return (math.sin((1.0 - alpha) * theta) / s) * q0 + \
           (math.sin(alpha         * theta) / s) * q1


# ---------------------------------------------------------------------------
# Entorno principal
# ---------------------------------------------------------------------------

class MobileManipulatorEnv(gym.Env):
    """
    Entorno Gymnasium sobre el robot WBC en Gazebo.

    Parámetros
    ----------
    host             : IP del rosbridge (default: localhost).
    port             : Puerto rosbridge (default: 9090).
    obstacle_models  : Lista de model_name de Gazebo usados como obstáculos.
    training_goals   : Lista de pose_goal para reset() aleatorio. Si es None,
                       usa DEFAULT_TRAINING_GOALS de config.py.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9090,
        obstacle_models: list = None,
        training_goals: list = None,
    ):
        super().__init__()

        self._ri = RobotInterface(host=host, port=port)
        self._obstacle_models  = obstacle_models or []
        self._training_goals   = training_goals or DEFAULT_TRAINING_GOALS

        # Espacios de acción y observación
        self.action_space = spaces.Discrete(6)

        low  = np.array([-1., -1., -1.,  0., -1., -1., 0., 0., 0.], dtype=np.float32)
        high = np.array([ 1.,  1.,  1.,  1.,  1.,  1., 1., 1., 1.], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Estado interno del episodio (inicializado en reset)
        self._pose_goal:    dict  = None   # {x,y,z,qx,qy,qz,qw,intent}
        self._goal_quat:    np.ndarray = None  # [qx,qy,qz,qw] del goal
        self._cmd_heading:  float = 0.0    # yaw de navegación (rad)
        self._prev_dist:    float = 0.0    # distancia al goal en el step anterior
        self._step_count:   int   = 0

        # Conectar al robot
        self._ri.connect()
        self._ri.wait_for_state(timeout=10.0)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """
        Reinicia el episodio.

        options (dict, opcional):
            pose_goal : dict con claves x,y,z,qx,qy,qz,qw,intent.
                        Si no se provee, se elige aleatoriamente de
                        self._training_goals.
        """
        super().reset(seed=seed)

        # 1. Seleccionar goal
        if options and "pose_goal" in options:
            self._pose_goal = options["pose_goal"]
        else:
            self._pose_goal = random.choice(self._training_goals)

        self._goal_quat = np.array([
            self._pose_goal["qx"], self._pose_goal["qy"],
            self._pose_goal["qz"], self._pose_goal["qw"],
        ], dtype=np.float32)

        # 2. Resetear Gazebo
        #    settle_time más largo: da tiempo al WBC de recibir el nuevo goal
        #    (home) antes de que el env empiece a leer observaciones.
        initial_state = self._ri.reset(settle_time=RESET_SETTLE_TIME, ee_home=_EE_HOME)

        # 3. Esperar estado válido post-reset (con reintentos)
        if initial_state is None:
            ok = self._ri.wait_for_state(timeout=5.0)
            if not ok:
                raise RuntimeError("No se recibió estado del robot tras reset(). "
                                   "Verificar que Gazebo y rosbridge estén activos.")
            initial_state = self._ri.get_state()

        # 3b. Sleep adicional para que el WBC llegue a home y estabilice
        #     antes de registrar la observación inicial.
        time.sleep(RESET_EXTRA_SLEEP)
        stable = self._ri.get_state()
        if stable is not None:
            initial_state = stable

        # 4. Inicializar heading a partir del yaw actual
        self._cmd_heading = initial_state["ee_rpy"][2]
        self._step_count  = 0

        # 5. Distancia inicial al goal
        ee = initial_state["ee_pos"]
        pg = self._pose_goal
        self._prev_dist = math.sqrt(
            (pg["x"] - ee[0])**2 +
            (pg["y"] - ee[1])**2 +
            (pg["z"] - ee[2])**2
        )

        obs  = self._compute_obs(initial_state)
        info = {"initial_dist": self._prev_dist, "pose_goal": self._pose_goal}
        return obs, info

    def step(self, action: int):
        """
        Ejecuta un step con la acción discreta dada.

        Returns: (obs, reward, terminated, truncated, info)
        """
        state = self._ri.get_state()
        if state is None:
            # Si no hay estado, aplicar penalización y truncar
            obs = np.zeros(9, dtype=np.float32)
            return obs, REWARD_STEP, False, True, {"reason": "no_state"}

        ee_pos = state["ee_pos"]
        ee_rpy = state["ee_rpy"]
        yaw    = ee_rpy[2]

        # ------------------------------------------------------------------
        # 1. Calcular Δpose según acción
        # ------------------------------------------------------------------
        dx = dy = dz = 0.0

        if action == ACTION_FORWARD:
            dx = STEP_XY * math.cos(self._cmd_heading)
            dy = STEP_XY * math.sin(self._cmd_heading)
        elif action == ACTION_TURN_LEFT:
            self._cmd_heading += TURN_ANGLE_RAD
        elif action == ACTION_TURN_RIGHT:
            self._cmd_heading -= TURN_ANGLE_RAD
        elif action == ACTION_MOVE_UP:
            dz = STEP_Z
        elif action == ACTION_MOVE_DOWN:
            dz = -STEP_Z
        # ACTION_STAY: sin movimiento

        # ------------------------------------------------------------------
        # 2. Nueva posición absoluta comandada — clampeada al workspace seguro
        # ------------------------------------------------------------------
        new_x = max(WORKSPACE_X[0], min(WORKSPACE_X[1], ee_pos[0] + dx))
        new_y = max(WORKSPACE_Y[0], min(WORKSPACE_Y[1], ee_pos[1] + dy))
        new_z = max(WORKSPACE_Z[0], min(WORKSPACE_Z[1], ee_pos[2] + dz))

        # ------------------------------------------------------------------
        # 3. Orientación: slerp desde orientación actual hacia goal_quat
        # ------------------------------------------------------------------
        actual_quat = _rpy_to_quat(*ee_rpy)
        cmd_quat    = _quat_slerp(actual_quat, self._goal_quat, SLERP_ALPHA)

        # ------------------------------------------------------------------
        # 4. Enviar goal al WBC y esperar
        # ------------------------------------------------------------------
        self._ri.send_goal(new_x, new_y, new_z, *cmd_quat)
        time.sleep(STEP_SLEEP)

        # ------------------------------------------------------------------
        # 5. Leer estado actualizado
        # ------------------------------------------------------------------
        new_state = self._ri.get_state()
        if new_state is None:
            new_state = state   # fallback al estado anterior

        # ------------------------------------------------------------------
        # 6. Calcular distancia al goal
        # ------------------------------------------------------------------
        pg         = self._pose_goal
        new_ee     = new_state["ee_pos"]
        dist_curr  = math.sqrt(
            (pg["x"] - new_ee[0])**2 +
            (pg["y"] - new_ee[1])**2 +
            (pg["z"] - new_ee[2])**2
        )

        # ------------------------------------------------------------------
        # 6b. Detección out-of-workspace (antes de consultar obstáculos)
        # ------------------------------------------------------------------
        out_of_ws = (
            not (WORKSPACE_X[0] <= new_ee[0] <= WORKSPACE_X[1]) or
            not (WORKSPACE_Y[0] <= new_ee[1] <= WORKSPACE_Y[1]) or
            not (WORKSPACE_Z[0] <= new_ee[2] <= WORKSPACE_Z[1])
        )

        # ------------------------------------------------------------------
        # 7. Sensor de obstáculos + colisión — UN solo recorrido WebSocket
        # ------------------------------------------------------------------
        new_yaw = new_state["ee_rpy"][2]
        front, left, right, collision = get_obstacle_info(
            self._ri._client,
            self._obstacle_models,
            new_ee,
            new_yaw,
        )

        # ------------------------------------------------------------------
        # 8. Calcular recompensa
        # ------------------------------------------------------------------
        reward  = self._prev_dist - dist_curr          # shaping: + si se acerca
        reward += REWARD_STEP
        if action == ACTION_STAY:
            reward += REWARD_STAY

        terminated = False
        if collision or out_of_ws:
            reward    += REWARD_COLLISION   # misma penalización que colisión
            terminated = True
        elif dist_curr < GOAL_THR:
            reward    += REWARD_SUCCESS
            terminated = True

        self._step_count += 1
        truncated = (self._step_count >= MAX_STEPS)
        self._prev_dist   = dist_curr

        # ------------------------------------------------------------------
        # 9. Observación (reutiliza front/left/right ya calculados)
        # ------------------------------------------------------------------
        obs  = self._compute_obs(new_state, front, left, right)
        info = {
            "dist_goal":  dist_curr,
            "collision":  collision,
            "success":    (dist_curr < GOAL_THR and not collision),
            "step_count": self._step_count,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self._ri.disconnect()
        except Exception:
            pass

    def render(self):
        pass   # No se implementa render (Gazebo ya tiene GUI)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _compute_obs(
        self, state: dict,
        front: float = 0.0, left: float = 0.0, right: float = 0.0,
    ) -> np.ndarray:
        """
        Construye el vector de observación de 9 componentes normalizado.
        Recibe front/left/right ya calculados por get_obstacle_info() en step()
        para no repetir las llamadas WebSocket. En reset() se llama sin ellos
        (obstáculos en 0 para la observación inicial).
        """
        ee  = state["ee_pos"]
        yaw = state["ee_rpy"][2]
        pg  = self._pose_goal

        dx_goal = pg["x"] - ee[0]
        dy_goal = pg["y"] - ee[1]
        dz_goal = pg["z"] - ee[2]
        dist    = math.sqrt(dx_goal**2 + dy_goal**2 + dz_goal**2)

        rel_angle = math.atan2(dy_goal, dx_goal) - yaw

        obs = np.array([
            dx_goal / MAX_DIST,
            dy_goal / MAX_DIST,
            dz_goal / MAX_DIST,
            dist    / MAX_DIST,
            math.sin(rel_angle),
            math.cos(rel_angle),
            front,
            left,
            right,
        ], dtype=np.float32)

        return np.clip(obs, self.observation_space.low, self.observation_space.high)
