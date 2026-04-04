"""
pybullet_sensor_utils.py — Máscara geométrica de obstáculos (PyBullet ground truth).

Reemplaza sensor_utils.py (que usaba /gazebo/get_model_state vía roslibpy).
La lógica de sectores angulares es idéntica; solo cambia la fuente de posición:
    sensor_utils:        roslibpy.Service → Gazebo
    pybullet_sensor_utils: PyBulletRobot.get_obstacle_position() → PyBullet

API pública compatible con el env:
    get_obstacle_info(robot, obstacle_models, ee_pos, heading_yaw_rad)
        → (front, left, right, collision)
"""

import math

from .config import (
    DETECTION_RADIUS, COLLISION_RADIUS,
    SECTOR_FRONT_HALF, SECTOR_SIDE_MAX,
)


def get_obstacle_info(
    robot,
    obstacle_models: list,
    ee_pos: list,
    heading_yaw_rad: float,
) -> tuple:
    """
    Recorre los obstáculos y devuelve proximidad sectorial continua + flag de colisión.

    Parámetros
    ----------
    robot           : instancia de PyBulletRobot con método get_obstacle_position().
    obstacle_models : lista de model_name a consultar (mismo string que en add_obstacle).
    ee_pos          : posición actual del efector [x, y, z].
    heading_yaw_rad : yaw del robot en radianes (de ee_rpy[2]).

    Retorna
    -------
    (front, left, right, collision)
        front/left/right : proximidad normalizada [0, 1].
                           0.0 = no hay obstáculo en el sector dentro de DETECTION_RADIUS.
                           1.0 = obstáculo a COLLISION_RADIUS o menos (inmediato).
                           Valores intermedios indican qué tan cerca está el obstáculo.
        collision        : bool — True si algún obstáculo < COLLISION_RADIUS (geométrico,
                           usado como respaldo; la colisión física es la fuente primaria).
    """
    if not obstacle_models:
        return 0.0, 0.0, 0.0, False

    rx, ry = ee_pos[0], ee_pos[1]
    front     = 0.0
    left      = 0.0
    right     = 0.0
    collision = False

    span = DETECTION_RADIUS - COLLISION_RADIUS   # rango útil de proximidad

    for model_name in obstacle_models:
        pos = robot.get_obstacle_position(model_name)
        if pos is None:
            continue

        dx   = pos[0] - rx
        dy   = pos[1] - ry
        dist = math.sqrt(dx * dx + dy * dy)

        # Colisión geométrica (respaldo)
        if dist < COLLISION_RADIUS:
            collision = True

        # Solo dentro del radio de detección
        if dist >= DETECTION_RADIUS:
            continue

        # Proximidad: 0 en el borde de detección, 1 en el radio de colisión
        proximity = max(0.0, min(1.0, (DETECTION_RADIUS - dist) / span))

        abs_angle_rad = math.atan2(dy, dx)
        rel_angle_deg = math.degrees(abs_angle_rad - heading_yaw_rad)

        # Normalizar a [-180, 180]
        while rel_angle_deg >  180.0:
            rel_angle_deg -= 360.0
        while rel_angle_deg < -180.0:
            rel_angle_deg += 360.0

        if abs(rel_angle_deg) < SECTOR_FRONT_HALF:
            front = max(front, proximity)
        elif SECTOR_FRONT_HALF <= rel_angle_deg <= SECTOR_SIDE_MAX:
            left = max(left, proximity)
        elif -SECTOR_SIDE_MAX <= rel_angle_deg <= -SECTOR_FRONT_HALF:
            right = max(right, proximity)

    return front, left, right, collision
