"""
sensor_utils.py — Máscara geométrica de obstáculos (ground truth Gazebo).

Consulta las posiciones de modelos en Gazebo vía rosbridge y determina si
algún obstáculo se encuentra en los sectores front/left/right del robot.

Consistente con la decisión de «omnisciencia de simulación» ya declarada
para el GoalBuilder: usamos /gazebo/get_model_state en lugar de LiDAR real.
"""

import math
import roslibpy

from .config import (
    DETECTION_RADIUS, COLLISION_RADIUS,
    SECTOR_FRONT_HALF, SECTOR_SIDE_MAX,
)

_SVC_MODEL_STATE = "/gazebo/get_model_state"
_SVC_TYPE        = "gazebo_msgs/GetModelState"


def get_obstacle_info(
    ros_client: roslibpy.Ros,
    obstacle_models: list,
    ee_pos: list,
    heading_yaw_rad: float,
) -> tuple:
    """
    Recorre los obstáculos UNA sola vez y devuelve máscara + flag de colisión.
    Evita duplicar las llamadas WebSocket entre _compute_obs y _check_collision.

    Args:
        ros_client      : instancia roslibpy.Ros ya conectada.
        obstacle_models : lista de model_name de Gazebo a consultar.
        ee_pos          : posición actual del efector [x, y, z].
        heading_yaw_rad : yaw del robot en radianes (de ee_rpy[2]).

    Returns:
        (front, left, right, collision)
            front/left/right : 0.0 o 1.0 — obstáculo detectado en ese sector
            collision        : bool — True si algún obstáculo < COLLISION_RADIUS
    """
    if not obstacle_models:
        return 0.0, 0.0, 0.0, False

    svc = roslibpy.Service(ros_client, _SVC_MODEL_STATE, _SVC_TYPE)
    rx, ry = ee_pos[0], ee_pos[1]

    front     = 0.0
    left      = 0.0
    right     = 0.0
    collision = False

    for model_name in obstacle_models:
        try:
            response = svc.call(roslibpy.ServiceRequest({
                "model_name":            model_name,
                "relative_entity_name":  "world",
            }))
        except Exception:
            continue

        if not response.get("success", False):
            continue

        pos  = response["pose"]["position"]
        dx   = pos["x"] - rx
        dy   = pos["y"] - ry
        dist = math.sqrt(dx * dx + dy * dy)

        # Colisión geométrica
        if dist < COLLISION_RADIUS:
            collision = True

        # Máscara de sectores (solo dentro del radio de detección)
        if dist >= DETECTION_RADIUS:
            continue

        abs_angle_rad = math.atan2(dy, dx)
        rel_angle_deg = math.degrees(abs_angle_rad - heading_yaw_rad)

        # Normalizar a [-180, 180]
        while rel_angle_deg >  180.0:
            rel_angle_deg -= 360.0
        while rel_angle_deg < -180.0:
            rel_angle_deg += 360.0

        if abs(rel_angle_deg) < SECTOR_FRONT_HALF:
            front = 1.0
        elif SECTOR_FRONT_HALF <= rel_angle_deg <= SECTOR_SIDE_MAX:
            left = 1.0
        elif -SECTOR_SIDE_MAX <= rel_angle_deg <= -SECTOR_FRONT_HALF:
            right = 1.0

    return front, left, right, collision
