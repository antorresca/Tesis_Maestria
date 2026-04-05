"""
arm_env.py — Entorno Gymnasium para el Agente 2 (posicionamiento Z del brazo).

El Agente 2 controla únicamente la coordenada Z del efector final.
Se activa tras el Agente 1 (que ya posicionó la base en XY).
Intents activos: pick, place.

Entrenamiento en aislamiento:
    - Base fija en posición home (XY = _EE_HOME[0], _EE_HOME[1]).
    - Z inicial = _EE_HOME[2]; z_goal muestreado en ARM_GOAL_SAMPLE_Z.
    - Sin obstáculos — el brazo solo sube/baja; las colisiones reales son raras
      en la configuración de home y se penalizan si ocurren (workspace_exit).

Observación (3 valores, normalizados):
    o = [
        dz / ARM_Z_SPAN,                          # error Z signed — dirección + magnitud
        (ee_z - WORKSPACE_Z[0]) / ARM_Z_SPAN,     # posición Z actual normalizada
        abs(dz) / ARM_Z_SPAN,                     # distancia Z absoluta
    ]

Acciones (Discrete(3)):
    0 = MOVE_UP   (+STEP_Z)
    1 = MOVE_DOWN (-STEP_Z)
    2 = STAY

Recompensa:
    r  = (|dz_prev| - |dz_curr|)   # shaping: + si se acerca a z_goal
    r += -0.01                      # step penalty
    r += -0.05                      # si action == STAY
    r += +20.0                      # éxito: |dz| < GOAL_Z_THR
    r += -20.0                      # out-of-workspace o colisión física

Compatibilidad: gymnasium >= 0.29  (SB3 >= 2.0)
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from ..robot_interface.pybullet_robot import PyBulletRobot
    from ..drl.config import (
        ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_ARM_STAY,
        STEP_Z, WORKSPACE_Z,
        REWARD_COLLISION, REWARD_SUCCESS, REWARD_STEP, REWARD_STAY,
        _EE_HOME,
    )
    from .config import (
        ARM_GOAL_SAMPLE_Z, GOAL_Z_THR, ARM_Z_SPAN, ARM_MAX_STEPS,
    )
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from robot_interface.pybullet_robot import PyBulletRobot
    from drl.config import (
        ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_ARM_STAY,
        STEP_Z, WORKSPACE_Z,
        REWARD_COLLISION, REWARD_SUCCESS, REWARD_STEP, REWARD_STAY,
        _EE_HOME,
    )
    from drl_agent2.config import (
        ARM_GOAL_SAMPLE_Z, GOAL_Z_THR, ARM_Z_SPAN, ARM_MAX_STEPS,
    )

# XY fijos durante el entrenamiento en aislamiento — posición home del EE
_TRAIN_X = _EE_HOME[0]
_TRAIN_Y = _EE_HOME[1]


class ArmEnv(gym.Env):
    """
    Entorno Gymnasium para el posicionamiento Z del brazo (Agente 2).

    Parámetros
    ----------
    gui      : True para abrir ventana gráfica de PyBullet.
    urdf_path: Ruta al URDF compilado. Si es None, usa el default.
    fixed_xy : (x, y) del EE durante el entrenamiento. Default: home position.
               En integración end-to-end, pasar la posición que dejó el Agente 1.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        gui: bool = False,
        urdf_path: str = None,
        fixed_xy: tuple = None,
    ):
        super().__init__()

        self._fixed_x = fixed_xy[0] if fixed_xy else _TRAIN_X
        self._fixed_y = fixed_xy[1] if fixed_xy else _TRAIN_Y

        self._ri = PyBulletRobot(urdf_path=urdf_path, gui=gui)

        # Acción: 3 acciones discretas (MOVE_UP, MOVE_DOWN, STAY)
        self.action_space = spaces.Discrete(3)

        # Observación: [dz_norm (signed), z_norm, dist_z_norm] — todos en [-1, 1] aprox.
        low  = np.array([-1., 0., 0.], dtype=np.float32)
        high = np.array([ 1., 1., 1.], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Estado interno del episodio
        self._z_goal:    float = 0.0
        self._prev_dz:   float = 0.0   # |dz| en el step anterior
        self._step_count: int  = 0

        self._ri.connect()
        self._ri.wait_for_state()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """
        Reinicia el episodio.

        options (dict, opcional):
            z_goal : float — altura Z objetivo del efector.
                     Si no se provee, se muestrea aleatoriamente de ARM_GOAL_SAMPLE_Z.
            fixed_xy: (x, y) — sobrescribir el XY fijo para este episodio.
        """
        super().reset(seed=seed)

        # 1. Actualizar XY fijo si se pasa en options
        if options and "fixed_xy" in options:
            self._fixed_x, self._fixed_y = options["fixed_xy"]

        # 2. Seleccionar z_goal
        if options and "z_goal" in options:
            self._z_goal = float(options["z_goal"])
        else:
            self._z_goal = float(np.random.uniform(*ARM_GOAL_SAMPLE_Z))

        # 3. Resetear simulación — home pose (joints=0, base en origen)
        initial_state = self._ri.reset(ee_home=_EE_HOME)
        if initial_state is None:
            raise RuntimeError("PyBulletRobot.reset() devolvió None inesperadamente.")

        # 4. Posicionar en XY fijo con z=home para iniciar desde postura conocida
        self._ri.send_goal(self._fixed_x, self._fixed_y, _EE_HOME[2],
                           0.0, 0.0, 0.0, 1.0)

        self._step_count = 0

        # 5. Leer estado real tras posicionamiento inicial
        state = self._ri.get_state()
        if state is None:
            raise RuntimeError("No se pudo obtener estado tras reset.")

        ee_z = state["ee_pos"][2]
        self._prev_dz = abs(self._z_goal - ee_z)

        obs  = self._compute_obs(state)
        info = {
            "z_goal":   self._z_goal,
            "initial_dz": self._prev_dz,
        }
        return obs, info

    def step(self, action: int):
        """
        Ejecuta un step con la acción discreta dada.

        Returns: (obs, reward, terminated, truncated, info)
        """
        state = self._ri.get_state()
        if state is None:
            obs = np.zeros(3, dtype=np.float32)
            return obs, REWARD_STEP, False, True, {"reason": "no_state"}

        ee_z = state["ee_pos"][2]

        # ------------------------------------------------------------------
        # 1. Calcular nueva Z según acción
        # ------------------------------------------------------------------
        if action == ACTION_MOVE_UP:
            new_z = ee_z + STEP_Z
        elif action == ACTION_MOVE_DOWN:
            new_z = ee_z - STEP_Z
        else:
            # ACTION_ARM_STAY
            new_z = ee_z

        # Clamp al workspace — si queda fuera, el out-of-ws lo detectamos abajo
        new_z = max(WORKSPACE_Z[0], min(WORKSPACE_Z[1], new_z))

        # ------------------------------------------------------------------
        # 2. Enviar goal — XY fijos, solo Z cambia; orientación upright
        # ------------------------------------------------------------------
        physics_collision = self._ri.send_goal(
            self._fixed_x, self._fixed_y, new_z,
            0.0, 0.0, 0.0, 1.0
        )

        # ------------------------------------------------------------------
        # 3. Leer estado actualizado
        # ------------------------------------------------------------------
        new_state = self._ri.get_state()
        if new_state is None:
            new_state = state

        new_ee_z = new_state["ee_pos"][2]
        dz_curr  = abs(self._z_goal - new_ee_z)

        # ------------------------------------------------------------------
        # 4. Out-of-workspace check
        # ------------------------------------------------------------------
        out_of_ws = not (WORKSPACE_Z[0] <= new_ee_z <= WORKSPACE_Z[1])

        # ------------------------------------------------------------------
        # 5. Calcular recompensa
        # ------------------------------------------------------------------
        reward  = self._prev_dz - dz_curr   # shaping: + si |dz| decrece
        reward += REWARD_STEP
        if action == ACTION_ARM_STAY:
            reward += REWARD_STAY

        terminated = False
        if physics_collision or out_of_ws:
            reward    += REWARD_COLLISION
            terminated = True
        elif dz_curr < GOAL_Z_THR:
            reward    += REWARD_SUCCESS
            terminated = True

        self._step_count += 1
        truncated       = (self._step_count >= ARM_MAX_STEPS)
        self._prev_dz   = dz_curr

        # ------------------------------------------------------------------
        # 6. Observación y métricas
        # ------------------------------------------------------------------
        obs  = self._compute_obs(new_state)
        info = {
            "dz":        self._z_goal - new_ee_z,   # signed, para análisis
            "dist_z":    dz_curr,
            "collision": physics_collision,
            "success":   (dz_curr < GOAL_Z_THR and not (physics_collision or out_of_ws)),
            "step_count": self._step_count,
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
    # Internos
    # ------------------------------------------------------------------

    def _compute_obs(self, state: dict) -> np.ndarray:
        """
        Construye el vector de observación de 3 componentes normalizado.

        [0] dz_norm  = (z_goal - ee_z) / ARM_Z_SPAN  — signed, [-1, 1] aprox.
        [1] z_norm   = (ee_z - Z_min)  / ARM_Z_SPAN  — posición actual, [0, 1]
        [2] dist_norm = |dz| / ARM_Z_SPAN             — distancia absoluta, [0, 1]
        """
        ee_z    = state["ee_pos"][2]
        dz      = self._z_goal - ee_z
        dz_norm = dz / ARM_Z_SPAN
        z_norm  = (ee_z - WORKSPACE_Z[0]) / ARM_Z_SPAN
        dist_norm = abs(dz) / ARM_Z_SPAN

        obs = np.array([dz_norm, z_norm, dist_norm], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)
