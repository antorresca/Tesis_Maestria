"""
continuous_env.py — Entorno Gymnasium con espacio de acción continuo para el Agente 1.

Extiende la lógica de MobileManipulatorEnv al espacio continuo Box(2).
La observación (9 valores) y la función de recompensa son idénticas al env
discreto para permitir comparación directa DQN vs SAC vs PPO.

Espacio de acción — Box(2), cada componente en [-1, 1]:
    action[0] = v_linear   — velocidad lineal a lo largo del cmd_heading
                             dx = v * STEP_XY * cos(heading)
                             dy = v * STEP_XY * sin(heading)
    action[1] = v_angular  — velocidad angular (yaw)
                             heading += omega * TURN_ANGLE_RAD

El robot es holonómico (Robotino3), pero se modela como diferencial en esta
fase para mantener la comparación limpia con el agente discreto.
Si se quiere aprovechar la omni-direccionalidad completa, extender a Box(3)
= [vx, vy, omega] en robot frame.

Observación (9 valores, normalizados) — idéntica al agente discreto:
    [dx_goal/MAX_DIST, dy_goal/MAX_DIST, dz_goal/MAX_DIST,
     dist/MAX_DIST, sin(rel_angle), cos(rel_angle),
     obstacle_front, obstacle_left, obstacle_right]

Diferencias respecto al env discreto:
    - action_space: Box(2) en vez de Discrete(5)
    - Sin penalización STAY (queda implícita en REWARD_STEP si v≈0)
    - Compatible con SAC y PPO continuo de SB3

Compatibilidad: gymnasium >= 0.29  (SB3 >= 2.0)
"""

import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from ..robot_interface.pybullet_robot import PyBulletRobot
    from .pybullet_sensor_utils import get_obstacle_info
    from .config import (
        STEP_XY, TURN_ANGLE_RAD, SLERP_ALPHA,
        MAX_DIST, GOAL_THR, COLLISION_RADIUS, DETECTION_RADIUS, MAX_STEPS,
        REWARD_COLLISION, REWARD_SUCCESS, REWARD_STEP, REWARD_PROXIMITY_SCALE,
        DEFAULT_TRAINING_GOALS, WORKSPACE_X, WORKSPACE_Y, WORKSPACE_Z,
        _EE_HOME, OBSTACLE_SPECS, SENSOR_OBSTACLE_MODELS,
        GOAL_SAMPLE_X, GOAL_SAMPLE_Y, GOAL_MIN_DIST, GOAL_MAX_TRIES,
    )
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from robot_interface.pybullet_robot import PyBulletRobot
    from drl.pybullet_sensor_utils import get_obstacle_info
    from drl.config import (
        STEP_XY, TURN_ANGLE_RAD, SLERP_ALPHA,
        MAX_DIST, GOAL_THR, COLLISION_RADIUS, DETECTION_RADIUS, MAX_STEPS,
        REWARD_COLLISION, REWARD_SUCCESS, REWARD_STEP, REWARD_PROXIMITY_SCALE,
        DEFAULT_TRAINING_GOALS, WORKSPACE_X, WORKSPACE_Y, WORKSPACE_Z,
        _EE_HOME, OBSTACLE_SPECS, SENSOR_OBSTACLE_MODELS,
        GOAL_SAMPLE_X, GOAL_SAMPLE_Y, GOAL_MIN_DIST, GOAL_MAX_TRIES,
    )


# ---------------------------------------------------------------------------
# Utilidades de cuaterniones (copiadas de mobile_manipulator_env para
# no crear dependencia circular — son funciones puras sin estado)
# ---------------------------------------------------------------------------

def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll  / 2), math.sin(roll  / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw   / 2), math.sin(yaw   / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dtype=np.float32)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1  = -q1
        dot = -dot
    dot = min(dot, 1.0)
    theta = math.acos(dot)
    if theta < 1e-6:
        return q0.copy()
    s = math.sin(theta)
    return (math.sin((1.0 - alpha) * theta) / s) * q0 + \
           (math.sin(alpha         * theta) / s) * q1


# ---------------------------------------------------------------------------
# Entorno continuo
# ---------------------------------------------------------------------------

class ContinuousManipulatorEnv(gym.Env):
    """
    Entorno Gymnasium con espacio de acción continuo Box(2) para el Agente 1.

    Parámetros
    ----------
    gui             : True para abrir ventana gráfica de PyBullet.
    obstacle_models : Lista de model_name a incluir como obstáculos.
    training_goals  : Lista de pose_goal para reset() aleatorio.
    urdf_path       : Ruta al URDF compilado.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        gui: bool = False,
        obstacle_models: list = None,
        training_goals: list = None,
        urdf_path: str = None,
    ):
        super().__init__()

        self._ri = PyBulletRobot(urdf_path=urdf_path, gui=gui)
        self._obstacle_models = obstacle_models or []
        self._training_goals  = training_goals or DEFAULT_TRAINING_GOALS

        # Espacio de acción: Box(2) — [v_linear, v_angular] ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observación: idéntica al agente discreto (9 valores)
        low  = np.array([-1., -1., -1.,  0., -1., -1., 0., 0., 0.], dtype=np.float32)
        high = np.array([ 1.,  1.,  1.,  1.,  1.,  1., 1., 1., 1.], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._pose_goal:   dict       = None
        self._goal_quat:   np.ndarray = None
        self._cmd_heading: float      = 0.0
        self._prev_dist:   float      = 0.0
        self._step_count:  int        = 0

        self._ri.connect()
        self._load_obstacles()
        self._ri.wait_for_state()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if options and "pose_goal" in options:
            self._pose_goal = options["pose_goal"]
        else:
            self._pose_goal = self._sample_random_goal()

        self._goal_quat = np.array([
            self._pose_goal["qx"], self._pose_goal["qy"],
            self._pose_goal["qz"], self._pose_goal["qw"],
        ], dtype=np.float32)

        initial_state = self._ri.reset(ee_home=_EE_HOME)
        if initial_state is None:
            raise RuntimeError("PyBulletRobot.reset() devolvió None inesperadamente.")

        self._cmd_heading = initial_state["ee_rpy"][2]
        self._step_count  = 0

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

    def step(self, action: np.ndarray):
        """
        Ejecuta un step con la acción continua dada.

        action : np.ndarray shape (2,)
            [0] v_linear  ∈ [-1, 1] — velocidad lineal (escalada por STEP_XY)
            [1] v_angular ∈ [-1, 1] — velocidad angular (escalada por TURN_ANGLE_RAD)

        Returns: (obs, reward, terminated, truncated, info)
        """
        state = self._ri.get_state()
        if state is None:
            obs = np.zeros(9, dtype=np.float32)
            return obs, REWARD_STEP, False, True, {"reason": "no_state"}

        v_lin   = float(np.clip(action[0], -1.0, 1.0))
        v_ang   = float(np.clip(action[1], -1.0, 1.0))

        ee_pos = state["ee_pos"]
        ee_rpy = state["ee_rpy"]

        # ------------------------------------------------------------------
        # 1. Calcular Δpose
        # ------------------------------------------------------------------
        dx = v_lin * STEP_XY * math.cos(self._cmd_heading)
        dy = v_lin * STEP_XY * math.sin(self._cmd_heading)
        self._cmd_heading += v_ang * TURN_ANGLE_RAD

        # ------------------------------------------------------------------
        # 2. Nueva posición clampeada al workspace
        # ------------------------------------------------------------------
        new_x = max(WORKSPACE_X[0], min(WORKSPACE_X[1], ee_pos[0] + dx))
        new_y = max(WORKSPACE_Y[0], min(WORKSPACE_Y[1], ee_pos[1] + dy))
        new_z = ee_pos[2]   # z fijo (Agente 1 no controla Z)

        # ------------------------------------------------------------------
        # 3. Orientación: slerp hacia goal_quat
        # ------------------------------------------------------------------
        actual_quat = _rpy_to_quat(*ee_rpy)
        cmd_quat    = _quat_slerp(actual_quat, self._goal_quat, SLERP_ALPHA)

        # ------------------------------------------------------------------
        # 4. Enviar goal — True si colisionó y revirtió
        # ------------------------------------------------------------------
        physics_collision = self._ri.send_goal(new_x, new_y, new_z, *cmd_quat)

        # ------------------------------------------------------------------
        # 5. Leer estado actualizado
        # ------------------------------------------------------------------
        new_state = self._ri.get_state()
        if new_state is None:
            new_state = state

        # ------------------------------------------------------------------
        # 6. Distancia al goal
        # ------------------------------------------------------------------
        pg        = self._pose_goal
        new_ee    = new_state["ee_pos"]
        dist_curr = math.sqrt(
            (pg["x"] - new_ee[0])**2 +
            (pg["y"] - new_ee[1])**2 +
            (pg["z"] - new_ee[2])**2
        )

        out_of_ws = (
            not (WORKSPACE_X[0] <= new_ee[0] <= WORKSPACE_X[1]) or
            not (WORKSPACE_Y[0] <= new_ee[1] <= WORKSPACE_Y[1]) or
            not (WORKSPACE_Z[0] <= new_ee[2] <= WORKSPACE_Z[1])
        )

        # ------------------------------------------------------------------
        # 7. Sensor de obstáculos
        # ------------------------------------------------------------------
        new_yaw = new_state["ee_rpy"][2]
        front, left, right, _ = get_obstacle_info(
            self._ri,
            SENSOR_OBSTACLE_MODELS,
            new_ee,
            new_yaw,
        )

        # ------------------------------------------------------------------
        # 8. Recompensa — idéntica al agente discreto (sin STAY penalty)
        # ------------------------------------------------------------------
        reward  = self._prev_dist - dist_curr
        reward += REWARD_STEP
        reward -= REWARD_PROXIMITY_SCALE * (front + 0.5 * max(left, right))

        terminated = False
        if physics_collision or out_of_ws:
            reward    += REWARD_COLLISION
            terminated = True
        elif dist_curr < GOAL_THR:
            reward    += REWARD_SUCCESS
            terminated = True

        self._step_count += 1
        truncated       = (self._step_count >= MAX_STEPS)
        self._prev_dist = dist_curr

        obs  = self._compute_obs(new_state, front, left, right)
        info = {
            "dist_goal":  dist_curr,
            "collision":  physics_collision,
            "success":    (dist_curr < GOAL_THR and not physics_collision),
            "step_count": self._step_count,
            "v_linear":   v_lin,
            "v_angular":  v_ang,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self._ri.disconnect()
        except Exception:
            pass

    def render(self):
        pass

    # ------------------------------------------------------------------
    # Internos — idénticos a MobileManipulatorEnv
    # ------------------------------------------------------------------

    def _sample_random_goal(self) -> dict:
        home_x, home_y = _EE_HOME[0], _EE_HOME[1]
        xmin, xmax = GOAL_SAMPLE_X
        ymin, ymax = GOAL_SAMPLE_Y
        margin = COLLISION_RADIUS + 0.05

        for _ in range(GOAL_MAX_TRIES):
            gx = float(np.random.uniform(xmin, xmax))
            gy = float(np.random.uniform(ymin, ymax))

            if math.sqrt((gx - home_x)**2 + (gy - home_y)**2) < GOAL_MIN_DIST:
                continue

            inside = False
            for spec in OBSTACLE_SPECS.values():
                ox, oy = spec["position"][0], spec["position"][1]
                hx, hy = spec["half_extents"][0], spec["half_extents"][1]
                if abs(gx - ox) < hx + margin and abs(gy - oy) < hy + margin:
                    inside = True
                    break
            if inside:
                continue

            return {
                "x": gx, "y": gy, "z": float(_EE_HOME[2]),
                "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                "intent": "navigate",
            }

        return random.choice(self._training_goals)

    def _load_obstacles(self):
        for model_name in self._obstacle_models:
            if model_name not in OBSTACLE_SPECS:
                print(f"[cont_env] WARN: '{model_name}' no está en OBSTACLE_SPECS — ignorado.")
                continue
            spec = OBSTACLE_SPECS[model_name]
            self._ri.add_obstacle(model_name, **spec)

    def _compute_obs(
        self, state: dict,
        front: float = 0.0, left: float = 0.0, right: float = 0.0,
    ) -> np.ndarray:
        ee  = state["ee_pos"]
        pg  = self._pose_goal

        dx_goal = pg["x"] - ee[0]
        dy_goal = pg["y"] - ee[1]
        dz_goal = pg["z"] - ee[2]
        dist    = math.sqrt(dx_goal**2 + dy_goal**2 + dz_goal**2)

        rel_angle = math.atan2(dy_goal, dx_goal) - self._cmd_heading

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
