# Estructura de la carpeta `data/`

Contiene todos los recursos del módulo Robot: el Docker, el workspace de ROS y scripts de prueba de comunicación.

---

## Jerarquía

```
data/
├── prueba_web.py                          # Script de prueba de comunicación via roslibpy (WebSocket → ROS)
└── docker/
    ├── ros1_docker/                       # Definición del contenedor Docker
    │   ├── Dockerfile                     # [agregar descripción]
    │   ├── docker-compose.yml             # [agregar descripción]
    │   └── CMakeLists.txt                 # [agregar descripción]
    └── ros1_workspace/                    # Catkin workspace montado dentro del Docker
        ├── src/
        │   ├── catkin/                    # Herramientas base de catkin (no modificar)
        │   │
        │   ├── manipulator_unal_description/       # Descripción URDF del brazo manipulador (INTERBOTIX VXSA-300)
        │   │   ├── urdf/                           # Archivos xacro/urdf del brazo
        │   │   ├── meshes/                         # Meshes STL de los eslabones del brazo
        │   │   ├── config/                         # Configuraciones de controladores (effort, group, pos)
        │   │   └── launch/                         # Launchers para visualizar y cargar el brazo
        │   │
        │   ├── mobile_robot_unal_description/      # Descripción URDF de la base móvil (Robotino3)
        │   │   ├── urdf/                           # Archivos xacro de la base
        │   │   ├── meshes/                         # Meshes .dae del Robotino3
        │   │   ├── src/                            # Nodos C++: odometría, EKF, corrección world-odom
        │   │   ├── config/                         # Params de controladores y EKF
        │   │   ├── launch/                         # Launchers Gazebo y visualización
        │   │   ├── rviz/                           # Configs de RViz (con y sin EKF)
        │   │   ├── rqt/                            # Configs rqt_multiplot para comparar posiciones
        │   │   ├── scripts/                        # Scripts Python de trayectorias de prueba
        │   │   └── worlds/                         # Mundos Gazebo (empty_world)
        │   │
        │   ├── mobile_manipulator_unal_description/  # Descripción del robot completo (base + brazo)
        │   │   ├── urdf/                             # URDF/xacro del sistema completo
        │   │   ├── config/                           # Params: EKF, controlador de esfuerzo del brazo, vel base, trayectoria
        │   │   ├── launch/                           # Launchers: spawn, gazebo, múltiples robots
        │   │   └── rviz/                             # Configs RViz para robot completo y múltiples
        │   │
        │   ├── mobile_manipulator_msgs/              # Mensajes ROS custom del sistema
        │   │   └── msg/
        │   │       ├── Trajectory.msg                # Pose destino + vel + accel + joints (interfaz principal con WBC)
        │   │       ├── MobileManipulator.msg         # Estado completo del robot
        │   │       ├── JointData.msg / Joints.msg    # Datos de articulaciones
        │   │       ├── Torques.msg                   # Torques calculados por el WBC
        │   │       ├── PositionData / OrientationData / RPY / Vector3 / SingularValues
        │   │
        │   ├── mob_manipulator_controller/           # WBC — Controlador de Cuerpo Completo (OSC)
        │   │   ├── src/
        │   │   │   ├── Mob_Manipulator_Controller.cpp  # Lógica principal del WBC
        │   │   │   ├── Stack_Tasks.cpp                 # Manejo de la jerarquía de tareas del WBC
        │   │   │   ├── Load_model.cpp                  # Carga del modelo dinámico (RBDL/Pinocchio)
        │   │   │   ├── osc_controller/                 # [agregar descripción]
        │   │   │   └── mob_manipulator_controller_node.cpp  # Nodo ROS del controlador
        │   │   ├── include/                            # Headers del WBC y OSC
        │   │   ├── config/config.yaml                  # Parámetros del controlador
        │   │   └── launch/osc_controller.launch        # Launcher del WBC
        │   │
        │   └── mobile_robot_vel_publisher/           # Publicador de velocidades para la base móvil
        │       ├── src/                              # Nodos C++: Global_Vel_Pub, Local_Vel_Pub
        │       ├── config/                           # Parámetros de velocidad global/local
        │       └── launch/                           # Launchers: global, local, robotino
```

---

## Módulos IA (`modules/`)

```
modules/
├── requirements.txt                        # deps base (roslibpy, numpy)
├── robot_interface/
│   └── robot_interface.py                  # puente roslibpy↔DRL
├── pln/
│   ├── requirements_pln.txt
│   ├── data/raw/                           # dataset semilla CSV + prompt LLM
│   ├── data/processed/                     # splits train/val/test
│   ├── models/                             # modelos fine-tuneados (gitignored)
│   └── src/
│       ├── prepare_dataset.py
│       ├── dataset.py
│       ├── train.py
│       ├── evaluate.py
│       ├── entity_extractor.py             # extractor reglas/regex (3 clases)
│       └── pln_module.py                   # interfaz: predict(text) → JSON
└── goal_builder/
    └── goal_builder.py                     # Fase 3: PLN JSON → pose_goal
                                            # consulta /gazebo/get_model_state
                                            # lookup table ENTITY_TO_GAZEBO
```

---

## Notas clave

- **Interfaz principal con el WBC:** topic `/mobile_manipulator/desired_traj` tipo `mobile_manipulator_msgs/Trajectory`
- **Articulaciones del robot:**
  - Base móvil: `mobjoint1`, `mobjoint2`, `mobjoint3` (Robotino3 — ruedas omnidireccionales)
  - Brazo: `joint1` a `joint6` (INTERBOTIX VXSA-300)
- **Comunicación externa → Docker:** via `roslibpy` sobre WebSocket (puerto `9090`, `rosbridge_server`)
- Los `.bag` en `ros1_workspace/` son grabaciones de sesiones de prueba del WBC base
