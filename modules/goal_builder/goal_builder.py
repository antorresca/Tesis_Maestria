"""
goal_builder.py
---------------
Fase 3: Convierte el JSON de salida del PLN en un pose_goal (xyz + cuaternión)
que consume el módulo DRL.

Pipeline:
    PLN JSON  →  GoalBuilder.build()  →  pose_goal dict
                      ↕
         /gazebo/get_model_state  (rosbridge WebSocket)

pose_goal format (compatible con robot_interface.send_goal):
    {
        'x': float, 'y': float, 'z': float,
        'qx': float, 'qy': float, 'qz': float, 'qw': float,
        'intent': str,
        'target': str | None,
        'destination': str | None,
    }

Uso como librería:
    gb = GoalBuilder(host='localhost', port=9090)
    gb.connect()
    pose_goal = gb.build({"intent": "navigate", "target": None,
                          "destination": "mesa", "confidence": 0.97})
    gb.disconnect()

    # Compartiendo cliente existente (e.g., desde RobotInterface):
    gb = GoalBuilder(ros_client=existing_ros_client)
    pose_goal = gb.build(pln_json)

Uso como script:
    python goal_builder.py --intent navigate --destination mesa
    python goal_builder.py --intent pick     --target "cubo azul"
    python goal_builder.py --intent place    --target cubo --destination bandeja
"""

import math
import json
from typing import Optional

import roslibpy


# ---------------------------------------------------------------------------
# Lookup table: nombre de entidad (salida entity_extractor) → model_name Gazebo
#
# Expandir / reemplazar según los modelos presentes en la escena actual.
# Las claves deben coincidir con la salida normalizada (lower, strip) del
# entity_extractor.  Los valores son los model_name tal como aparecen en
# /gazebo/get_model_state.
# ---------------------------------------------------------------------------
ENTITY_TO_GAZEBO: dict = {
    # --- Objetos manipulables ---
    "cubo":           "red_cube",   # único cubo en training_world
    "cube":           "red_cube",
    "caja":           "unit_box",
    "box":            "unit_box",
    "cubo azul":      "blue_cube",
    "blue cube":      "blue_cube",
    "cubo rojo":      "red_cube",
    "red cube":       "red_cube",
    "cubo verde":     "green_cube",
    "green cube":     "green_cube",
    "cilindro":       "unit_cylinder",
    "cylinder":       "unit_cylinder",
    "botella":        "bottle",
    "bottle":         "bottle",
    "bloque":         "unit_box",
    "block":          "unit_box",
    # --- Destinos / zonas ---
    "mesa":           "table",
    "table":          "table",
    "bandeja":        "tray",
    "tray":           "tray",
    "zona verde":     "green_zone",
    "green zone":     "green_zone",
    "zona roja":      "red_zone",
    "red zone":       "red_zone",
    "zona azul":      "blue_zone",
    "blue zone":      "blue_zone",
    "zona amarilla":  "yellow_zone",
    "yellow zone":    "yellow_zone",
    "cocina":         "kitchen_area",
    "kitchen":        "kitchen_area",
    "workspace":      "workspace",
    "área de trabajo": "workspace",
    "area de trabajo": "workspace",
}

# ---------------------------------------------------------------------------
# Pose de spawn del robot → destino "home"
# ---------------------------------------------------------------------------
HOME_POSE = {
    "x": 0.0, "y": 0.0, "z": 0.3,
    "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
}

# ---------------------------------------------------------------------------
# Alturas de referencia (m) — calibradas para training_world.world
#
#   Z_TRAVEL: libre de obstáculos (0.5 m) con margen → 0.60 m
#   Z_PICK  : 5 cm sobre el cubo rojo (centro z=0.775) → 0.85 m
#   Z_PLACE : 10 cm sobre la bandeja (z=0.02) → 0.12 m
#
# Ajustar si se cambia la escena Gazebo.
# ---------------------------------------------------------------------------
Z_TRAVEL = 0.60  # altura segura de desplazamiento del efector final
Z_PICK   = 0.85  # altura de agarre (sobre el objeto en la mesa)
Z_PLACE  = 0.12  # altura de depósito (sobre la bandeja en el suelo)

# ---------------------------------------------------------------------------
# Distancia de standoff para navigate (m)
# El robot se detiene frente al objetivo a esta distancia.
# ---------------------------------------------------------------------------
NAVIGATE_STANDOFF = 0.6

# ---------------------------------------------------------------------------
# Orientaciones predefinidas (qx, qy, qz, qw)
#
# ORIENT_NEUTRAL:  efector nivelado (identidad), estable para navegación.
# ORIENT_TOP_DOWN: efector apuntando hacia -Z, rotación de +90° alrededor
#                  del eje Y.  Usado para pick/place desde arriba.
#                  Ajustar si la convención del frame del efector difiere.
# ---------------------------------------------------------------------------
ORIENT_NEUTRAL  = (0.0, 0.0,   0.0,   1.0)
ORIENT_TOP_DOWN = (0.0, 0.707, 0.0, 0.707)


class GoalBuilder:
    """
    Convierte el output del PLN en un pose_goal Cartesiano consultando el
    estado de los modelos en Gazebo vía rosbridge.

    Parámetros
    ----------
    host        : IP del rosbridge (ignorado si se provee ros_client).
    port        : Puerto del rosbridge (ignorado si se provee ros_client).
    ros_client  : Instancia roslibpy.Ros ya conectada (para compartir la
                  conexión con RobotInterface y evitar abrir dos sockets).
    entity_map  : Override del lookup table ENTITY_TO_GAZEBO (opcional).
    """

    _SVC_MODEL_STATE = "/gazebo/get_model_state"
    _SVC_TYPE        = "gazebo_msgs/GetModelState"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9090,
        ros_client: Optional[roslibpy.Ros] = None,
        entity_map: Optional[dict] = None,
    ):
        self._own_client = ros_client is None
        self._client     = roslibpy.Ros(host=host, port=port) if self._own_client else ros_client
        self._entity_map = entity_map if entity_map is not None else ENTITY_TO_GAZEBO

    # ------------------------------------------------------------------
    # Conexión
    # ------------------------------------------------------------------

    def connect(self):
        """Conecta al rosbridge. Solo necesario cuando NO se pasó ros_client."""
        if self._own_client:
            self._client.run()

    def disconnect(self):
        """Desconecta. Solo actúa si el cliente es propio."""
        if self._own_client and self._client.is_connected:
            self._client.terminate()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def build(self, pln_json: dict) -> dict:
        """
        Construye el pose_goal a partir del JSON del PLN.

        Args:
            pln_json: {"intent": str, "target": str|None,
                       "destination": str|None, "confidence": float}

        Returns:
            pose_goal dict con claves:
                x, y, z, qx, qy, qz, qw, intent, target, destination

        Raises:
            ValueError:  intent desconocido o entidad no encontrada en la tabla.
            RuntimeError: el modelo Gazebo no existe o no está cargado.
        """
        intent      = pln_json.get("intent")
        target      = pln_json.get("target")
        destination = pln_json.get("destination")

        if intent == "navigate":
            return self._build_navigate(destination)
        elif intent == "pick":
            return self._build_pick(target)
        elif intent == "place":
            return self._build_place(target, destination)
        else:
            raise ValueError(f"Intent desconocido: '{intent}'.")

    # ------------------------------------------------------------------
    # Builders por intent
    # ------------------------------------------------------------------

    def _build_navigate(self, destination: Optional[str]) -> dict:
        if destination is None:
            raise ValueError("navigate requiere 'destination'.")

        if destination == "home":
            return {**HOME_POSE, "intent": "navigate", "target": None, "destination": "home"}

        model_name  = self._resolve_entity(destination)
        model_pose  = self._get_model_pose(model_name)

        # Standoff: posicionar el robot frente al objetivo en el plano XY.
        # El ángulo apunta desde el origen (spawn) hacia el objeto.
        angle = math.atan2(model_pose["y"], model_pose["x"])
        x = model_pose["x"] - NAVIGATE_STANDOFF * math.cos(angle)
        y = model_pose["y"] - NAVIGATE_STANDOFF * math.sin(angle)

        # Orientación: yaw hacia el objetivo (rotación en Z)
        qz = math.sin(angle / 2.0)
        qw = math.cos(angle / 2.0)

        return {
            "x": x, "y": y, "z": Z_TRAVEL,
            "qx": 0.0, "qy": 0.0, "qz": qz, "qw": qw,
            "intent": "navigate",
            "target": None,
            "destination": destination,
        }

    def _build_pick(self, target: Optional[str]) -> dict:
        if target is None:
            raise ValueError("pick requiere 'target'.")

        model_name = self._resolve_entity(target)
        model_pose = self._get_model_pose(model_name)
        qx, qy, qz, qw = ORIENT_TOP_DOWN

        return {
            "x": model_pose["x"], "y": model_pose["y"], "z": Z_PICK,
            "qx": qx, "qy": qy, "qz": qz, "qw": qw,
            "intent": "pick",
            "target": target,
            "destination": None,
        }

    def _build_place(self, target: Optional[str], destination: Optional[str]) -> dict:
        if destination is None:
            raise ValueError("place requiere 'destination'.")

        model_name = self._resolve_entity(destination)
        model_pose = self._get_model_pose(model_name)
        qx, qy, qz, qw = ORIENT_TOP_DOWN

        return {
            "x": model_pose["x"], "y": model_pose["y"], "z": Z_PLACE,
            "qx": qx, "qy": qy, "qz": qz, "qw": qw,
            "intent": "place",
            "target": target,
            "destination": destination,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_entity(self, entity: str) -> str:
        """Normaliza la entidad y la mapea al model_name de Gazebo."""
        key = entity.lower().strip()
        model_name = self._entity_map.get(key)
        if model_name is None:
            raise ValueError(
                f"Entidad '{entity}' no encontrada en ENTITY_TO_GAZEBO. "
                f"Agregar la entrada correspondiente al scene de Gazebo."
            )
        return model_name

    def _get_model_pose(self, model_name: str) -> dict:
        """Llama a /gazebo/get_model_state y retorna la posición {x, y, z}."""
        svc = roslibpy.Service(self._client, self._SVC_MODEL_STATE, self._SVC_TYPE)
        request = roslibpy.ServiceRequest({
            "model_name": model_name,
            "relative_entity_name": "world",
        })
        response = svc.call(request)

        if not response.get("success", False):
            raise RuntimeError(
                f"Gazebo no encontró el modelo '{model_name}'. "
                f"Status: {response.get('status_message', 'unknown')}"
            )

        pos = response["pose"]["position"]
        return {"x": pos["x"], "y": pos["y"], "z": pos["z"]}


# ---------------------------------------------------------------------------
# CLI de prueba
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GoalBuilder — prueba CLI")
    parser.add_argument("--host",        default="localhost")
    parser.add_argument("--port",        type=int, default=9090)
    parser.add_argument("--intent",      required=True, choices=["navigate", "pick", "place"])
    parser.add_argument("--target",      default=None)
    parser.add_argument("--destination", default=None)
    parser.add_argument("--confidence",  type=float, default=1.0)
    args = parser.parse_args()

    pln_json = {
        "intent":      args.intent,
        "target":      args.target,
        "destination": args.destination,
        "confidence":  args.confidence,
    }

    gb = GoalBuilder(host=args.host, port=args.port)
    gb.connect()
    try:
        pose_goal = gb.build(pln_json)
        print(json.dumps(pose_goal, indent=2))
    finally:
        gb.disconnect()
