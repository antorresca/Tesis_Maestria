"""
config.py — Hiperparámetros centralizados para el módulo DRL.

Todos los valores numéricos del entorno, recompensas y algoritmos SB3
se definen aquí para que el resto del módulo importe desde un único lugar.
"""

import math

# ---------------------------------------------------------------------------
# Acciones (índices)
# ---------------------------------------------------------------------------
ACTION_FORWARD    = 0
ACTION_TURN_LEFT  = 1
ACTION_TURN_RIGHT = 2
ACTION_MOVE_UP    = 3
ACTION_MOVE_DOWN  = 4
ACTION_STAY       = 5

ACTION_NAMES = {
    ACTION_FORWARD:    "forward",
    ACTION_TURN_LEFT:  "turn_left",
    ACTION_TURN_RIGHT: "turn_right",
    ACTION_MOVE_UP:    "move_up",
    ACTION_MOVE_DOWN:  "move_down",
    ACTION_STAY:       "stay",
}

# ---------------------------------------------------------------------------
# Dinámica del robot — pasos por acción
# ---------------------------------------------------------------------------
STEP_XY       = 0.05              # m — desplazamiento XY por step
STEP_Z        = 0.03              # m — desplazamiento Z por step
TURN_ANGLE    = 10.0              # grados — rotación yaw por step
TURN_ANGLE_RAD = math.radians(TURN_ANGLE)

# Interpolación de orientación (slerp hacia goal_quat cada step)
SLERP_ALPHA   = 0.1               # fracción de avance por step

# ---------------------------------------------------------------------------
# Entorno
# ---------------------------------------------------------------------------
MAX_DIST         = 5.0    # m — distancia máxima para normalización
GOAL_THR         = 0.25   # m — umbral de éxito (más amplio para primeros entrenamientos)
COLLISION_RADIUS = 0.25   # m — radio de colisión con obstáculos
DETECTION_RADIUS = 1.5    # m — radio de detección de obstáculos
MAX_STEPS        = 300    # pasos máximos — episodios más cortos = más diversidad

# Workspace seguro — el EE no puede salir de este cubo.
# Si sale → episodio termina como fallo (evita la explosión numérica de Gazebo).
# Ajustar según el tamaño real de training_world.world.
WORKSPACE_X = (-2.5,  2.5)   # m
WORKSPACE_Y = (-2.5,  2.5)   # m
WORKSPACE_Z = ( 0.05, 1.2)   # m

# Tiempo de espera después de send_goal antes de leer el estado
# (~921 Hz headless → ~1 ms por ciclo WBC; 0.1 s da ~92 ciclos de margen)
STEP_SLEEP       = 0.05   # s  (~46 ciclos WBC a 921 Hz — suficiente para Δ 5cm)

# Tiempo de estabilización en reset:
#   - robot_interface.reset() hace reset_world + sleep + send_goal(home)
#   - RESET_SETTLE_TIME: cuánto espera robot_interface antes de enviar home
#   - RESET_EXTRA_SLEEP:  tiempo adicional que espera el env tras recibir
#     el estado inicial, para que el WBC llegue a la posición home
RESET_SETTLE_TIME  = 0.8   # s — pasado a robot_interface.reset()
RESET_EXTRA_SLEEP  = 0.5   # s — sleep extra en env.reset() post-estabilización

# ---------------------------------------------------------------------------
# Sensor de obstáculos — sectores angulares (grados)
# ---------------------------------------------------------------------------
SECTOR_FRONT_HALF = 45.0    # frente: |rel_angle| < 45°
SECTOR_SIDE_MAX   = 135.0   # lateral: 45° a 135° (izq) o -45° a -135° (der)

# ---------------------------------------------------------------------------
# Obstáculos estáticos en training_world.world
# Rellenar con los model_name tal como aparecen en Gazebo.
# Dejar vacío si se entrena sin obstáculos (prueba base).
# ---------------------------------------------------------------------------
TRAINING_OBSTACLE_MODELS: list = [
    "obstacle_1",
    "obstacle_2",
    "table",   # la mesa tiene presencia física — incluir para evasión
]

# ---------------------------------------------------------------------------
# Goals de entrenamiento por defecto (pose_goal format)
# Usados en reset() cuando no se proporciona pose_goal vía options.
# ---------------------------------------------------------------------------
# Goals cortos para entrenamiento inicial — ACTUALIZAR con posición real del EE en home.
# El robot spawna en (0,0,0) pero el EE no está en (0,0,0).
# Paso 1: correr `rostopic echo /mobile_manipulator/data -n 1` tras reset
#         y anotar current_position (x_home, y_home, z_home).
# Paso 2: estos goals deben estar a 0.4–1.2 m del home EE, no del origen.
#
# PLACEHOLDER — reemplazar x_h/y_h/z_h con los valores reales del EE home.
# Por ahora se usan deltas relativos al origen asumiendo EE home ≈ (0.3, 0, 0.4).
_EE_HOME = (0.3, 0.0, 0.4)   # ← AJUSTAR tras medir con rostopic

DEFAULT_TRAINING_GOALS = [
    # navigate corto — 0.5–1.0 m del EE home, misma altura Z (sin cambio vertical)
    {"x": _EE_HOME[0]+0.6, "y": _EE_HOME[1]+0.0, "z": _EE_HOME[2],
     "qx": 0.0, "qy": 0.0, "qz": 0.0,    "qw": 1.0,   "intent": "navigate"},
    {"x": _EE_HOME[0]-0.6, "y": _EE_HOME[1]+0.0, "z": _EE_HOME[2],
     "qx": 0.0, "qy": 0.0, "qz": 1.0,    "qw": 0.0,   "intent": "navigate"},
    {"x": _EE_HOME[0]+0.0, "y": _EE_HOME[1]+0.6, "z": _EE_HOME[2],
     "qx": 0.0, "qy": 0.0, "qz": 0.707,  "qw": 0.707, "intent": "navigate"},
    {"x": _EE_HOME[0]+0.0, "y": _EE_HOME[1]-0.6, "z": _EE_HOME[2],
     "qx": 0.0, "qy": 0.0, "qz":-0.707,  "qw": 0.707, "intent": "navigate"},
    # navigate diagonal
    {"x": _EE_HOME[0]+0.5, "y": _EE_HOME[1]+0.5, "z": _EE_HOME[2],
     "qx": 0.0, "qy": 0.0, "qz": 0.383,  "qw": 0.924, "intent": "navigate"},
    {"x": _EE_HOME[0]+0.5, "y": _EE_HOME[1]-0.5, "z": _EE_HOME[2],
     "qx": 0.0, "qy": 0.0, "qz":-0.383,  "qw": 0.924, "intent": "navigate"},
]

# ---------------------------------------------------------------------------
# Recompensas
# ---------------------------------------------------------------------------
REWARD_COLLISION = -10.0
REWARD_SUCCESS   =  20.0
REWARD_STEP      =  -0.01
REWARD_STAY      =  -0.05   # penalización adicional sobre REWARD_STEP si stay

# ---------------------------------------------------------------------------
# Hiperparámetros SB3 — DQN
# ---------------------------------------------------------------------------
DQN_HPARAMS = {
    "learning_rate":            1e-4,
    "buffer_size":              100_000,
    "learning_starts":          1_000,
    "batch_size":               64,
    "tau":                      1.0,
    "gamma":                    0.99,
    "train_freq":               4,
    "gradient_steps":           1,
    "target_update_interval":   500,
    "exploration_fraction":     0.2,
    "exploration_final_eps":    0.05,
    "policy_kwargs":            {"net_arch": [128, 128]},
}

# ---------------------------------------------------------------------------
# Hiperparámetros SB3 — PPO
# ---------------------------------------------------------------------------
PPO_HPARAMS = {
    "learning_rate":  3e-4,
    "n_steps":        2048,
    "batch_size":     64,
    "n_epochs":       10,
    "gamma":          0.99,
    "gae_lambda":     0.95,
    "clip_range":     0.2,
    "ent_coef":       0.01,
    "policy_kwargs":  {"net_arch": [128, 128]},
}

# Timesteps totales de entrenamiento (ajustar según tiempo disponible)
TOTAL_TIMESTEPS = 500_000
