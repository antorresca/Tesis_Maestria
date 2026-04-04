"""
config.py — Hiperparámetros centralizados para el módulo DRL (PyBullet).

Todos los valores numéricos del entorno, recompensas y algoritmos SB3
se definen aquí para que el resto del módulo importe desde un único lugar.
"""

import math

# ---------------------------------------------------------------------------
# Acciones (índices)
# ---------------------------------------------------------------------------
# Agente 1 — Navegación XY (base)
ACTION_FORWARD    = 0
ACTION_TURN_LEFT  = 1
ACTION_TURN_RIGHT = 2
ACTION_BACKWARD   = 3
ACTION_STAY       = 4

ACTION_NAMES = {
    ACTION_FORWARD:    "forward",
    ACTION_TURN_LEFT:  "turn_left",
    ACTION_TURN_RIGHT: "turn_right",
    ACTION_BACKWARD:   "backward",
    ACTION_STAY:       "stay",
}

# Agente 2 — Posicionamiento Z (brazo) — definido en arm_env.py (pendiente)
ACTION_MOVE_UP    = 0
ACTION_MOVE_DOWN  = 1
ACTION_ARM_STAY   = 2

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
MAX_STEPS        = 500    # pasos máximos — aumentado para goals lejanos/detrás del robot

# Workspace seguro — el EE no puede salir de este cubo.
# Si sale → episodio termina como fallo (evita la explosión numérica de Gazebo).
# Ajustar según el tamaño real de training_world.world.
WORKSPACE_X = (-2.5,  2.5)   # m
WORKSPACE_Y = (-2.5,  2.5)   # m
WORKSPACE_Z = ( 0.35, 1.20)  # m — Agente 1 mantiene z=home; Agente 2 controla z

# PyBullet es síncrono — no se necesitan sleeps entre send_goal y get_state.
# STEP_SLEEP / RESET_SETTLE_TIME / RESET_EXTRA_SLEEP eliminados (eran ROS/Gazebo).

# ---------------------------------------------------------------------------
# Sensor de obstáculos — sectores angulares (grados)
# ---------------------------------------------------------------------------
SECTOR_FRONT_HALF = 45.0    # frente: |rel_angle| < 45°
SECTOR_SIDE_MAX   = 135.0   # lateral: 45° a 135° (izq) o -45° a -135° (der)

# ---------------------------------------------------------------------------
# Obstáculos estáticos en PyBullet — training_world equivalente
# Cada entrada define la geometría y posición del cuerpo para PyBulletRobot.add_obstacle().
#   position     : [x, y, z] centro del cuerpo
#   half_extents : [hx, hy, hz] para caja (usada si se especifica)
# ---------------------------------------------------------------------------
OBSTACLE_SPECS: dict = {
    "obstacle_1": {
        "position":    [0.7,  0.7, 0.25],   # y: borde interior en 0.55; pasillo entre obs1/obs2 = 1.1m
        "half_extents": [0.15, 0.15, 0.25],
    },
    "obstacle_2": {
        "position":    [0.7, -0.7, 0.25],   # y: -0.5→-0.7
        "half_extents": [0.15, 0.15, 0.25],
    },
    "table": {
        "position":    [2.0,  0.0, 0.05],
        "half_extents": [0.40, 0.30, 0.05],   # plataforma baja 0.8×0.6×0.10 m
    },
}

TRAINING_OBSTACLE_MODELS: list = list(OBSTACLE_SPECS.keys())

# Modelos que el sensor de proximidad monitorea durante entrenamiento y evaluación.
# La tabla es el DESTINO de pick/place — no es un obstáculo de navegación.
# Incluirla en el sensor genera señal de "peligro" justo cuando el robot llega
# al goal, causando oscilación. Solo obstacle_1/2 son peligros reales de navegación.
SENSOR_OBSTACLE_MODELS: list = ['obstacle_1', 'obstacle_2']

# ---------------------------------------------------------------------------
# Muestreo aleatorio de goals — Agente 1
# En cada episodio se muestrea un goal (x, y) uniformemente dentro de este
# rectángulo, rechazando puntos dentro de obstáculos o demasiado cerca del spawn.
# Esto fuerza generalización: el agente no puede memorizar trayectorias fijas.
# DEFAULT_TRAINING_GOALS se usa solo como fallback si el sampler falla.
# ---------------------------------------------------------------------------
GOAL_SAMPLE_X  = (-1.5,  2.3)   # rango X del EE goal (m)
GOAL_SAMPLE_Y  = (-1.5,  1.5)   # rango Y del EE goal (m)
GOAL_MIN_DIST  =  0.50           # dist. mínima al EE home para evitar goals triviales (m)
GOAL_MAX_TRIES =  200            # intentos antes de usar fallback (lista fija)

# ---------------------------------------------------------------------------
# Goals de entrenamiento por defecto (pose_goal format)
# Usados en reset() cuando no se proporciona pose_goal vía options.
# ---------------------------------------------------------------------------
# EE home medido via PyBullet FK con joints=0 (2026-04-03).
# Nota: en Gazebo se medía 0.536 — la diferencia (~6 cm) es por el frame
# de referencia del EE en DART vs PyBullet. El valor PyBullet es el canónico.
_EE_HOME = (0.472, 0.000, 0.747)   # posición del efector en spawn (0,0,0), joints=0

def _g(dx, dy):
    """Shorthand: goal relativo al EE home, z=home fijo (Agente 1 — navegación XY)."""
    return {"x": _EE_HOME[0]+dx, "y": _EE_HOME[1]+dy, "z": _EE_HOME[2],
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0, "intent": "navigate"}

# Agente 1: goals solo en XY — z siempre home
DEFAULT_TRAINING_GOALS = [
    # adelante (pasa entre obstáculos en x≈0.7, y=±0.7)
    _g(+0.6,  0.0),
    _g(+0.9,  0.0),
    _g(+1.5,  0.0),
    # detrás (giro ~180°)
    _g(-0.5,  0.0),
    _g(-1.0,  0.0),
    _g(-1.0, +0.5),
    _g(-1.0, -0.5),
    # laterales
    _g( 0.0, +0.8),
    _g( 0.0, +1.2),
    _g( 0.0, -0.8),
    _g( 0.0, -1.2),
    # diagonales bordeando obstáculos
    _g(+0.5, +0.9),
    _g(+0.5, -0.9),
    _g(+1.2, +0.9),
    _g(+1.2, -0.9),
    _g(+1.5, +0.5),
    _g(+1.5, -0.5),
]

# ---------------------------------------------------------------------------
# Recompensas
# ---------------------------------------------------------------------------
REWARD_COLLISION       = -20.0   # -10→-20: colisión debe costar más que el shaping acumulado
REWARD_SUCCESS         =  20.0
REWARD_STEP            =  -0.01
REWARD_STAY            =  -0.05  # penalización adicional sobre REWARD_STEP si stay
# Penalización continua de proximidad: reward -= SCALE * (front + 0.5 * max(left, right))
# Con SCALE=0.20: front=1.0 → -0.20/step > shaping de distancia → fuerza desvío antes del choque.
# Reemplaza la penalización binaria anterior (REWARD_PROXIMITY = -0.05 cuando front > 0.5).
REWARD_PROXIMITY_SCALE =   0.10

# ---------------------------------------------------------------------------
# Hiperparámetros SB3 — DQN
# ---------------------------------------------------------------------------
DQN_HPARAMS = {
    "learning_rate":            1e-4,
    "buffer_size":              200_000,   # 100k→200k: más experiencia diversa en replay buffer
    "learning_starts":          2_000,
    "batch_size":               128,       # 64→128: gradientes más estables
    "tau":                      1.0,
    "gamma":                    0.99,
    "train_freq":               4,
    "gradient_steps":           1,
    "target_update_interval":   1_000,    # 500→1000: target network más estable
    # exploration_fraction relativo a TOTAL_TIMESTEPS (500k).
    # 0.1 → exploración termina en step 50k → 450k steps de explotación.
    # Experimento 2 usó 0.2 → solo 20k explotación antes de early stop.
    # Experimento 3 usó 0.1 → 450k explotación pero plateau en óptimo local.
    "exploration_fraction":     0.15,     # 0.1→0.15: termina en 75k, más exploración inicial
    "exploration_final_eps":    0.10,
    "policy_kwargs":            {"net_arch": [256, 128]},  # [128,128]→[256,128]: más capacidad
}

# ---------------------------------------------------------------------------
# Hiperparámetros SB3 — PPO
# ---------------------------------------------------------------------------
PPO_HPARAMS = {
    "learning_rate":  3e-4,
    "n_steps":        4096,   # 2048→4096: más transiciones por update con goals aleatorios
    "batch_size":     128,    # 64→128: igual que DQN para comparación justa
    "n_epochs":       10,
    "gamma":          0.99,
    "gae_lambda":     0.95,
    "clip_range":     0.2,
    "ent_coef":       0.01,   # entropía: fuerza exploración (crítico con goals aleatorios)
    "policy_kwargs":  {"net_arch": [256, 128]},  # misma arquitectura que DQN para fair comparison
}

# Timesteps totales de entrenamiento (ajustar según tiempo disponible)
TOTAL_TIMESTEPS = 1_000_000
