# Contexto del Proyecto — Tesis de Profundización

> Leer este archivo al inicio de cada sesión para recuperar el contexto completo sin releer todo el código.

---

## ¿Qué es este proyecto?

Sistema multi-agente para control de un **robot manipulador móvil** (base Robotino3 + brazo INTERBOTIX VXSA-300) simulado en **Gazebo/ROS Melodic** dentro de un contenedor **Docker**. El robot ya existe y está controlado por un **WBC (Whole-Body Controller)** basado en OSC (Operational Space Control) a **1000 Hz**. El trabajo de tesis consiste en añadir capas de IA encima del robot existente.

---

## Tecnologías descartadas

- **ROS2 Humble:** Migración intentada pero descartada. El WBC bajó de 1000 Hz a 100-200 Hz — inaceptable para el control.
- **CoppeliaSim:** Descartado por latencia que dañaba el rendimiento del controlador.

---

## Arquitectura (4 módulos)

```
Usuario → [PLN] → [Task Planning] → [DRL] → [Robot/WBC en Docker]
                        ↑
               [Visión de Máquina]
```

| Módulo | Tecnología | Estado |
|---|---|---|
| PLN | BERT o similar (balance costo/precisión) | Por desarrollar |
| Task Planning | Planificador jerárquico simple (HRL = plus, no requerido) | Por desarrollar |
| DRL | PPO vs SAC (comparar métricas) | Por desarrollar |
| Robot + WBC | ROS Melodic + Gazebo + Docker | **Existe y funciona** |

**Principio clave:** Arquitectura modular — cada módulo es reemplazable sin afectar los demás. E.g., migrar Task Planning simple a HRL no debe romper PLN ni DRL.

---

## Robot existente (Módulo Robot)

- **Base:** Robotino3 (ruedas omnidireccionales) — joints: `mobjoint1`, `mobjoint2`, `mobjoint3`
- **Brazo:** INTERBOTIX VXSA-300 — joints: `joint1` a `joint6`
- **Controlador:** WBC/OSC en C++ (`mob_manipulator_controller`) — **1000 Hz**
- **Spawn:** posición (0, 0, 0) mirando +X en Gazebo

### Modos del WBC (`Stack_Tasks.cpp`)

| Modo | DOF usados | Descripción |
|---|---|---|
| `mobile_manipulator` | 9 (base + brazo) | Coordinación completa para alcanzar pose |
| `only_manipulator` | 6 (solo brazo) | Solo el brazo alcanza posición/orientación |
| `mobile_robot` | 3 (solo base) | Solo la base navega a posición |

- Velocidad: **constante** (`ConstVel`) o **dinámica** (full dynamics)
- Selección actual: hardcodeada en comentarios de `Stack_Tasks.cpp` → **pendiente parametrizar en runtime**

### Topics ROS

| Topic | Tipo | Dirección |
|---|---|---|
| `/mobile_manipulator/desired_traj` | `mobile_manipulator_msgs/Trajectory` | Entrada al WBC (comando) |
| `/mobile_manipulator/data` | `mobile_manipulator_msgs/MobileManipulator` | Salida del WBC (estado completo) |
| `/mobile_manipulator/joint_states` | `sensor_msgs/JointState` | Estado de articulaciones |
| `/mobile_manipulator/commands/velocity` | `geometry_msgs/Twist` | Velocidad de la base |

### Problema identificado: Arm Overextension

- Enviar objetivo en +X (dirección de spawn) causa extensión completa del brazo **sin mover la base**
- Causa: `AchieveJointConf()` solo corre en `cycle==1` (posible bug), no hay penalización por extensión
- **Pendiente resolver en Fase 1**

---

## Comunicación entre módulos

- **Protocolo:** WebSocket — `roslibpy` + `rosbridge_server` (puerto `9090`)
- **Estado:** Comunicación validada con `data/prueba_web.py` ✓
- **Importante:** `rosbridge_server` **NO está en los launch files actuales** — hay que agregarlo
- Los módulos IA corren **fuera del Docker**; el robot corre **dentro**
- Docker usa `network_mode: host` → todos los puertos del container disponibles en localhost

---

## Interfaces entre módulos (definidas en plan)

```
PLN → Task Planning:   JSON { intent, target, destination, confidence }
Task Planning → DRL:   JSON { task_id, mode, objective{pose}, constraints, subtask_index }
DRL → Robot:           /mobile_manipulator/desired_traj (Trajectory msg)
Robot → DRL:           /mobile_manipulator/data (MobileManipulator msg)
```

---

## Paquetes ROS relevantes

| Paquete | Contenido |
|---|---|
| `mobile_manipulator_msgs` | Mensajes custom: `Trajectory`, `MobileManipulator`, `Torques`, etc. |
| `mob_manipulator_controller` | WBC/OSC — controlador principal (C++) |
| `mobile_manipulator_unal_description` | URDF completo + launch files del sistema |
| `mobile_robot_unal_description` | URDF + odometría + EKF de la base Robotino3 |
| `manipulator_unal_description` | URDF + meshes del brazo VXSA-300 |
| `test_trajectories_osc` | Scripts Python de prueba de trayectorias con el WBC |

---

## Archivos críticos

| Archivo | Relevancia |
|---|---|
| [CLAUDE.md](../CLAUDE.md) | Guía maestra del proyecto para Claude |
| [data/data.md](../data/data.md) | Jerarquía comentada de la carpeta data/ |
| [data/prueba_web.py](../data/prueba_web.py) | Referencia de comunicación roslibpy↔rosbridge |
| `data/docker/ros1_docker/Dockerfile` | Imagen del robot (ROS Melodic + DART + Gazebo) |
| `data/docker/ros1_docker/docker-compose.yml` | Configuración Docker (network: host, GPU) |
| `src/mob_manipulator_controller/src/Stack_Tasks.cpp` | Modos WBC, arm overextension, jerarquía de tareas |
| `src/mob_manipulator_controller/src/Mob_Manipulator_Controller.cpp` | Loop de control principal (1000 Hz) |
| `src/mobile_manipulator_unal_description/launch/general_launch.launch` | Launch principal — **agregar rosbridge aquí** |
| `src/mob_manipulator_controller/config/config.yaml` | Ganancias OSC y parámetros configurables |

---

## Plan de desarrollo (fases)

```
Fase 0: Infraestructura (rosbridge, documentar arm overextension, parametrizar modos)
    └── Fase 1: Robot interface limpia (robot_interface.py, resolver overextension)
            ├── Fase 2: PLN (BERT → JSON)              ─┐
            ├── Fase 3: Task Planning (JSON → subtareas) ├─ paralelas
            └── Fase 4: DRL (PPO vs SAC, env Gym)       ─┘
                    └── Fase 5: Integración end-to-end
```

---

## Estado actual (actualizar al avanzar)

- [x] Robot base existente y funcionando en Docker (1000 Hz)
- [x] Comunicación WebSocket probada (roslibpy → rosbridge)
- [x] Plan de desarrollo definido y aprobado
- [ ] Fase 0: Infraestructura — rosbridge en launch files, documentar arm overextension
- [ ] Fase 1: robot_interface.py + resolver arm overextension
- [ ] Fase 2: Módulo PLN
- [ ] Fase 3: Módulo Task Planning
- [ ] Fase 4: Módulo DRL (PPO vs SAC)
- [ ] Fase 5: Integración end-to-end

---

## Decisiones abiertas

- ¿Módulo de Visión de Máquina se desarrolla en esta tesis o se usa mock?
- ¿Alcance del DRL? ¿Solo base, solo brazo, o coordinado?
- ¿Task Planning genera comandos en espacio Cartesiano o de configuración?
- ¿La comparación PPO vs SAC necesita publicación o es solo para la tesis?
