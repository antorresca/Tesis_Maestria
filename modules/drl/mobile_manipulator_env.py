"""
mobile_manipulator_env.py — Entorno Gymnasium para el robot manipulador móvil.

Implementa la interfaz gym.Env sobre PyBulletRobot + pybullet_sensor_utils.
El DRL elige una acción discreta (Δpose incremental) y recibe:
    - Observación: vector de 9 valores normalizados.
    - Recompensa:  shaping de distancia + colisión + éxito + penalizaciones.

Uso típico (training loop):
    env = MobileManipulatorEnv(obstacle_models=['obstacle_1', 'table'])
    obs, info = env.reset(options={'pose_goal': pose_goal_dict})
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()

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
        ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
        ACTION_BACKWARD, ACTION_STAY,
        STEP_XY, STEP_Z, TURN_ANGLE_RAD, SLERP_ALPHA,
        MAX_DIST, GOAL_THR, COLLISION_RADIUS, DETECTION_RADIUS, MAX_STEPS,
        REWARD_COLLISION, REWARD_SUCCESS, REWARD_STEP, REWARD_STAY, REWARD_PROXIMITY_SCALE,
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
        ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
        ACTION_BACKWARD, ACTION_STAY,
        STEP_XY, STEP_Z, TURN_ANGLE_RAD, SLERP_ALPHA,
        MAX_DIST, GOAL_THR, COLLISION_RADIUS, DETECTION_RADIUS, MAX_STEPS,
        REWARD_COLLISION, REWARD_SUCCESS, REWARD_STEP, REWARD_STAY, REWARD_PROXIMITY_SCALE,
        DEFAULT_TRAINING_GOALS, WORKSPACE_X, WORKSPACE_Y, WORKSPACE_Z,
        _EE_HOME, OBSTACLE_SPECS, SENSOR_OBSTACLE_MODELS,
        GOAL_SAMPLE_X, GOAL_SAMPLE_Y, GOAL_MIN_DIST, GOAL_MAX_TRIES,
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
    Entorno Gymnasium sobre el robot en PyBullet.

    Parámetros
    ----------
    gui             : True para abrir ventana gráfica de PyBullet.
    obstacle_models : Lista de model_name a incluir como obstáculos.
                      Deben estar definidos en OBSTACLE_SPECS (config.py).
    training_goals  : Lista de pose_goal para reset() aleatorio. Si es None,
                      usa DEFAULT_TRAINING_GOALS de config.py.
    urdf_path       : Ruta al URDF compilado. Si es None, usa el default.
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
        self._obstacle_models  = obstacle_models or []
        self._training_goals   = training_goals or DEFAULT_TRAINING_GOALS

        # Espacios de acción y observación — Agente 1: 5 acciones XY
        self.action_space = spaces.Discrete(5)

        low  = np.array([-1., -1., -1.,  0., -1., -1., 0., 0., 0.], dtype=np.float32)
        high = np.array([ 1.,  1.,  1.,  1.,  1.,  1., 1., 1., 1.], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Estado interno del episodio (inicializado en reset)
        self._pose_goal:    dict  = None   # {x,y,z,qx,qy,qz,qw,intent}
        self._goal_quat:    np.ndarray = None  # [qx,qy,qz,qw] del goal
        self._cmd_heading:  float = 0.0    # yaw de navegación (rad)
        self._prev_dist:    float = 0.0    # distancia al goal en el step anterior
        self._step_count:   int   = 0

        # Inicializar PyBullet y cargar obstáculos
        self._ri.connect()
        self._load_obstacles()
        self._ri.wait_for_state()

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
            self._pose_goal = self._sample_random_goal()

        self._goal_quat = np.array([
            self._pose_goal["qx"], self._pose_goal["qy"],
            self._pose_goal["qz"], self._pose_goal["qw"],
        ], dtype=np.float32)

        # 2. Resetear simulación PyBullet (síncrono — sin sleeps)
        initial_state = self._ri.reset(ee_home=_EE_HOME)

        if initial_state is None:
            raise RuntimeError("PyBulletRobot.reset() devolvió None inesperadamente.")

        # 3. Inicializar heading a partir del yaw actual
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
        elif action == ACTION_BACKWARD:
            dx = -STEP_XY * math.cos(self._cmd_heading)
            dy = -STEP_XY * math.sin(self._cmd_heading)
        elif action == ACTION_TURN_LEFT:
            self._cmd_heading += TURN_ANGLE_RAD
        elif action == ACTION_TURN_RIGHT:
            self._cmd_heading -= TURN_ANGLE_RAD
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
        # 4. Enviar goal al robot — devuelve True si colisionó y revirtió
        # ------------------------------------------------------------------
        physics_collision = self._ri.send_goal(new_x, new_y, new_z, *cmd_quat)

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
        # 7. Sensor de obstáculos — sectores angulares para la observación.
        #    La colisión real ya fue detectada por send_goal (PyBullet contacts).
        #    Se usa SENSOR_OBSTACLE_MODELS (solo obs1/obs2): la tabla es el destino
        #    de pick/place, no un peligro de navegación; incluirla en el sensor
        #    genera señal de "peligro" al acercarse al goal y provoca oscilación.
        # ------------------------------------------------------------------
        new_yaw = new_state["ee_rpy"][2]
        front, left, right, _ = get_obstacle_info(
            self._ri,
            SENSOR_OBSTACLE_MODELS,
            new_ee,
            new_yaw,
        )
        collision = physics_collision

        # ------------------------------------------------------------------
        # 8. Calcular recompensa
        # ------------------------------------------------------------------
        reward  = self._prev_dist - dist_curr          # shaping: + si se acerca
        reward += REWARD_STEP
        if action == ACTION_STAY:
            reward += REWARD_STAY
        # Penalización continua de proximidad: escala con la proximidad real.
        # front ∈ [0,1]; flancos pesan 0.5x (menos críticos que frente).
        # Cuando front≈1: -0.20/step > shaping de distancia → fuerza desvío.
        reward -= REWARD_PROXIMITY_SCALE * (front + 0.5 * max(left, right))

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
        pass   # Render via GUI de PyBullet si gui=True en el constructor

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _sample_random_goal(self) -> dict:
        """
        Muestrea un goal aleatorio dentro de GOAL_SAMPLE_X × GOAL_SAMPLE_Y,
        rechazando posiciones que caigan dentro de cualquier obstáculo (con margen
        de COLLISION_RADIUS) o demasiado cerca del spawn (< GOAL_MIN_DIST).

        Si tras GOAL_MAX_TRIES intentos no se encuentra un punto válido, usa
        random.choice(self._training_goals) como fallback (no debería ocurrir).
        """
        home_x, home_y = _EE_HOME[0], _EE_HOME[1]
        xmin, xmax = GOAL_SAMPLE_X
        ymin, ymax = GOAL_SAMPLE_Y
        margin = COLLISION_RADIUS + 0.05   # holgura extra sobre el radio de colisión

        for _ in range(GOAL_MAX_TRIES):
            gx = float(np.random.uniform(xmin, xmax))
            gy = float(np.random.uniform(ymin, ymax))

            # Descartar goals trivialmente cerca del spawn
            if math.sqrt((gx - home_x) ** 2 + (gy - home_y) ** 2) < GOAL_MIN_DIST:
                continue

            # Descartar goals dentro del bounding box de cualquier obstáculo
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

        # Fallback: nunca debería llegar aquí con GOAL_MAX_TRIES=200
        return random.choice(self._training_goals)

    def _load_obstacles(self):
        """Crea los cuerpos de obstáculos en PyBullet según OBSTACLE_SPECS."""
        for model_name in self._obstacle_models:
            if model_name not in OBSTACLE_SPECS:
                print(f"[env] WARN: '{model_name}' no está en OBSTACLE_SPECS — ignorado.")
                continue
            spec = OBSTACLE_SPECS[model_name]
            self._ri.add_obstacle(model_name, **spec)

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
