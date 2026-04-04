"""
pybullet_robot.py
-----------------
Simulador kinématico del robot manipulador móvil en PyBullet.

Reemplaza robot_interface.py (ROS/Gazebo) con la misma API pública:
    connect(), disconnect(), get_state(), wait_for_state(),
    send_goal(x, y, z, qx, qy, qz, qw), reset(settle_time, ee_home)

Estrategia de control (kinematic — sin dinámica ROS/WBC):
  - Base holonómica: posición/yaw seteada directamente con
    resetBasePositionAndOrientation.
  - Brazo (6 DOF): ángulos via calculateInverseKinematics + resetJointState.
  - No se usa stepSimulation para la cinemática; solo para que PyBullet
    actualice la caché de FK antes de getLinkState.

Requisitos:
    pip install pybullet
"""

import math
import os
import pybullet as p
import pybullet_data


# ---------------------------------------------------------------------------
# Constantes cinemáticas (medidas via PyBullet FK con joints=0, spawn en origen)
# ---------------------------------------------------------------------------
_ARM_EE_OFFSET_X = 0.472   # m — EE delante de la base en heading del robot (joints=0)
_ARM_EE_OFFSET_Y = 0.000   # m — EE lateral respecto a la base (joints=0)
_ARM_EE_OFFSET_Z = 0.747   # m — EE altura desde el suelo (joints=0)

# Nombre URDF del link del efector final
_EE_LINK_NAME  = "mobile_manipulator/end_effector_link"

# Joints activos del brazo (en orden de aparición en el URDF)
_ARM_JOINT_NAMES = [
    "waist_joint",
    "shoulder_joint",
    "elbow_joint",
    "upper_forearm_joint",
    "lower_forearm_joint",
    "wrist_joint",
]

# Joints de la base (continuous — ruedas) — no se controlan activamente
_WHEEL_JOINT_NAMES = ["wheel0_joint", "wheel1_joint", "wheel2_joint"]

# URDF por defecto (relativo a este archivo)
_DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "urdf", "mobile_manipulator_pybullet.urdf",
)


class PyBulletRobot:
    """
    Simulador kinématico del robot en PyBullet.

    Parámetros
    ----------
    urdf_path   : ruta al URDF compilado (default: urdf/mobile_manipulator_pybullet.urdf)
    gui         : True para abrir ventana gráfica, False para headless (DIRECT)
    ik_max_iter : iteraciones máximas del solver IK de PyBullet
    """

    def __init__(
        self,
        urdf_path: str = None,
        gui: bool = False,
        ik_max_iter: int = 200,
    ):
        self._urdf_path  = urdf_path or _DEFAULT_URDF
        self._gui        = gui
        self._ik_max_iter = ik_max_iter

        self._client     = None   # physicsClient ID
        self._robot_id   = None   # bodyUniqueId del robot
        self._plane_id   = None   # bodyUniqueId del plano
        self._obstacle_ids: dict = {}  # model_name → bodyUniqueId

        # Índices de joints y links (poblados en _build_index)
        self._joint_name_to_idx: dict = {}
        self._arm_joint_indices:  list = []
        self._ee_link_idx:        int  = -1

        # Pose de base interna (x, y, yaw) — se actualiza en cada send_goal
        self._base_x   = 0.0
        self._base_y   = 0.0
        self._base_yaw = 0.0

    # ------------------------------------------------------------------
    # Conexión
    # ------------------------------------------------------------------

    def connect(self):
        """Inicializa PyBullet, carga el plano y el robot."""
        mode = p.GUI if self._gui else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)

        # Plano de suelo
        self._plane_id = p.loadURDF(
            "plane.urdf",
            physicsClientId=self._client,
        )

        # Robot con base libre (móvil)
        self._robot_id = p.loadURDF(
            self._urdf_path,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self._client,
        )

        self._build_index()

        # Desactivar motores de ruedas (no los controlamos)
        for joint_idx in [self._joint_name_to_idx[n]
                          for n in _WHEEL_JOINT_NAMES
                          if n in self._joint_name_to_idx]:
            p.setJointMotorControl2(
                self._robot_id, joint_idx,
                p.VELOCITY_CONTROL, force=0,
                physicsClientId=self._client,
            )

    def disconnect(self):
        """Cierra la conexión con PyBullet."""
        if self._client is not None:
            try:
                p.disconnect(self._client)
            except Exception:
                pass
            self._client = None

    @property
    def connected(self):
        return self._client is not None

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    def get_state(self):
        """
        Devuelve el estado actual del robot.

        Retorna
        -------
        dict con claves:
            ee_pos  : [x, y, z]           posición del efector final (m)
            ee_rpy  : [roll, pitch, yaw]  orientación del efector (rad)
            joints  : dict nombre→ángulo  (rad) para todos los joints
        """
        if self._client is None or self._robot_id is None:
            return None

        link_state = p.getLinkState(
            self._robot_id, self._ee_link_idx,
            computeForwardKinematics=True,
            physicsClientId=self._client,
        )
        ee_pos  = list(link_state[4])   # worldLinkFramePosition
        ee_orn  = link_state[5]          # worldLinkFrameOrientation (quaternion)
        ee_rpy  = list(p.getEulerFromQuaternion(ee_orn))

        # Joints del brazo
        joints = {}
        for name, idx in self._joint_name_to_idx.items():
            js = p.getJointState(self._robot_id, idx, physicsClientId=self._client)
            joints[name] = js[0]

        return {
            "ee_pos": ee_pos,
            "ee_rpy": ee_rpy,
            "joints": joints,
        }

    def wait_for_state(self, timeout: float = 5.0) -> bool:
        """Siempre True — el sim es síncrono, el estado está disponible inmediatamente."""
        return True

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def send_goal(
        self,
        x: float, y: float, z: float,
        qx: float = 0.0, qy: float = 0.0,
        qz: float = 0.0, qw: float = 1.0,
    ) -> bool:
        """
        Mueve el robot para que el EE quede en (x, y, z) con orientación (qx,qy,qz,qw).

        Estrategia kinematic:
          1. Extrae yaw del cuaternión.
          2. Posiciona la base de modo que el EE quede en (x, y) con el brazo home.
          3. Resuelve IK del brazo para afinar z y la orientación.
          4. Aplica directamente con resetBasePositionAndOrientation + resetJointState.
          5. Verifica colisión real con PyBullet; revierte si hay contacto con obstáculo.

        Retorna
        -------
        True si el movimiento fue bloqueado por colisión (posición revertida), False si OK.
        """
        # Guardar pose anterior para poder revertir
        prev_base_pos, prev_base_orn = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._client,
        )
        prev_joint_angles = [
            p.getJointState(self._robot_id, idx, physicsClientId=self._client)[0]
            for idx in self._arm_joint_indices
        ]
        prev_base_x, prev_base_y, prev_base_yaw = self._base_x, self._base_y, self._base_yaw

        # 1. Yaw objetivo del robot
        _, _, target_yaw = p.getEulerFromQuaternion([qx, qy, qz, qw])
        self._base_yaw = target_yaw

        # 2. Posición base para que EE quede en (x, y) con brazo home
        cy, sy = math.cos(target_yaw), math.sin(target_yaw)
        self._base_x = x - _ARM_EE_OFFSET_X * cy + _ARM_EE_OFFSET_Y * sy
        self._base_y = y - _ARM_EE_OFFSET_X * sy - _ARM_EE_OFFSET_Y * cy

        # 3. Aplicar pose de base
        base_pos = [self._base_x, self._base_y, 0.0]
        base_orn = p.getQuaternionFromEuler([0.0, 0.0, target_yaw])
        p.resetBasePositionAndOrientation(
            self._robot_id, base_pos, base_orn,
            physicsClientId=self._client,
        )

        # 4. IK del brazo para ajustar z (y residual xy)
        joint_angles = p.calculateInverseKinematics(
            self._robot_id,
            self._ee_link_idx,
            targetPosition=[x, y, z],
            targetOrientation=[qx, qy, qz, qw],
            maxNumIterations=self._ik_max_iter,
            physicsClientId=self._client,
        )

        # 5. Aplicar solo los joints del brazo (IK devuelve todos los joints no-fijos)
        num_joints = p.getNumJoints(self._robot_id, physicsClientId=self._client)
        movable_joints = [
            j for j in range(num_joints)
            if p.getJointInfo(self._robot_id, j, physicsClientId=self._client)[2]
            != p.JOINT_FIXED
        ]
        arm_set = set(self._arm_joint_indices)
        for i, joint_idx in enumerate(movable_joints):
            if joint_idx in arm_set and i < len(joint_angles):
                p.resetJointState(
                    self._robot_id, joint_idx, joint_angles[i],
                    physicsClientId=self._client,
                )

        # 6. Detectar colisión real con PyBullet — solo contra obstáculos registrados
        obstacle_body_ids = {info[0] for info in self._obstacle_ids.values()}
        p.performCollisionDetection(physicsClientId=self._client)
        contacts = p.getContactPoints(
            bodyA=self._robot_id,
            physicsClientId=self._client,
        )
        collided = any(
            c[2] in obstacle_body_ids
            for c in (contacts or [])
        )

        if collided:
            # Revertir a la pose anterior
            p.resetBasePositionAndOrientation(
                self._robot_id, prev_base_pos, prev_base_orn,
                physicsClientId=self._client,
            )
            for idx, angle in zip(self._arm_joint_indices, prev_joint_angles):
                p.resetJointState(
                    self._robot_id, idx, angle,
                    physicsClientId=self._client,
                )
            self._base_x, self._base_y, self._base_yaw = prev_base_x, prev_base_y, prev_base_yaw

        return collided

    # ------------------------------------------------------------------
    # Reset de episodio
    # ------------------------------------------------------------------

    def reset(self, settle_time: float = None, ee_home=None):
        """
        Resetea el robot a la posición spawn (origen, joints=0).

        Parámetros
        ----------
        settle_time : ignorado (compatibilidad con robot_interface.py)
        ee_home     : tuple (x, y, z) — si se provee, envía send_goal al home
                      tras resetear joints.

        Retorna
        -------
        Initial state dict (mismo formato que get_state()).
        """
        if self._client is None:
            return None

        # Resetear base a origen
        p.resetBasePositionAndOrientation(
            self._robot_id,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self._client,
        )
        self._base_x = self._base_y = self._base_yaw = 0.0

        # Resetear todos los joints a 0
        for idx in self._arm_joint_indices:
            p.resetJointState(
                self._robot_id, idx, 0.0,
                physicsClientId=self._client,
            )

        # Resetear obstáculos a sus posiciones iniciales
        for model_name, info in self._obstacle_ids.items():
            body_id, init_pos, init_orn = info
            p.resetBasePositionAndOrientation(
                body_id, init_pos, init_orn,
                physicsClientId=self._client,
            )

        # Mover brazo a posición home si se especifica
        if ee_home is not None:
            hx, hy, hz = ee_home
            self.send_goal(hx, hy, hz, 0.0, 0.0, 0.0, 1.0)

        return self.get_state()

    # ------------------------------------------------------------------
    # Gestión de obstáculos
    # ------------------------------------------------------------------

    def add_obstacle(
        self,
        model_name: str,
        position: list,
        half_extents: list = None,
        radius: float = None,
        height: float = None,
        orientation: list = None,
    ):
        """
        Crea un obstáculo estático en la simulación.

        Parámetros
        ----------
        model_name   : identificador (mismo que TRAINING_OBSTACLE_MODELS en config.py)
        position     : [x, y, z] del centro
        half_extents : [hx, hy, hz] para cuerpo tipo caja (AABB)
        radius       : radio para cilindro o esfera
        height       : altura para cilindro
        orientation  : [qx, qy, qz, qw] (default: identidad)
        """
        orn = orientation or [0.0, 0.0, 0.0, 1.0]

        if half_extents is not None:
            col_id = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half_extents,
                physicsClientId=self._client,
            )
            vis_id = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half_extents,
                rgbaColor=[0.8, 0.2, 0.2, 1.0],
                physicsClientId=self._client,
            )
        elif radius is not None and height is not None:
            col_id = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=radius, height=height,
                physicsClientId=self._client,
            )
            vis_id = p.createVisualShape(
                p.GEOM_CYLINDER, radius=radius, length=height,
                rgbaColor=[0.8, 0.5, 0.2, 1.0],
                physicsClientId=self._client,
            )
        elif radius is not None:
            col_id = p.createCollisionShape(
                p.GEOM_SPHERE, radius=radius,
                physicsClientId=self._client,
            )
            vis_id = p.createVisualShape(
                p.GEOM_SPHERE, radius=radius,
                rgbaColor=[0.5, 0.8, 0.2, 1.0],
                physicsClientId=self._client,
            )
        else:
            raise ValueError("Especificar half_extents, o radius (+ height para cilindro).")

        body_id = p.createMultiBody(
            baseMass=0,                         # estático
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position,
            baseOrientation=orn,
            physicsClientId=self._client,
        )
        self._obstacle_ids[model_name] = (body_id, list(position), list(orn))
        return body_id

    def get_obstacle_position(self, model_name: str):
        """
        Retorna la posición [x, y, z] de un obstáculo, o None si no existe.
        Equivalente a /gazebo/get_model_state.
        """
        if model_name not in self._obstacle_ids:
            return None
        body_id = self._obstacle_ids[model_name][0]
        pos, _ = p.getBasePositionAndOrientation(
            body_id, physicsClientId=self._client,
        )
        return list(pos)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _build_index(self):
        """Construye mapeos nombre→índice de joint y encuentra el link EE."""
        num_joints = p.getNumJoints(self._robot_id, physicsClientId=self._client)
        for j in range(num_joints):
            info = p.getJointInfo(self._robot_id, j, physicsClientId=self._client)
            joint_name = info[1].decode("utf-8")
            link_name  = info[12].decode("utf-8")
            self._joint_name_to_idx[joint_name] = j

            if link_name == _EE_LINK_NAME:
                self._ee_link_idx = j

        if self._ee_link_idx == -1:
            raise RuntimeError(
                f"No se encontró el link '{_EE_LINK_NAME}' en el URDF. "
                "Verificar que el URDF sea el correcto."
            )

        # Índices de joints del brazo (en el orden definido en _ARM_JOINT_NAMES)
        self._arm_joint_indices = [
            self._joint_name_to_idx[name]
            for name in _ARM_JOINT_NAMES
            if name in self._joint_name_to_idx
        ]
