# Contexto del Proyecto — Tesis de Profundización

> Leer al inicio de cada sesión. Ver `data/data.md` para jerarquía de archivos Docker/ROS.

## Proyecto

Robot manipulador móvil (base **Robotino3** + brazo **INTERBOTIX VXSA-300**) simulado en **Gazebo/ROS Melodic** dentro de **Docker**. Controlado por **WBC/OSC** a ~921 Hz en simulación. La tesis agrega módulos de IA encima del robot existente. Hardware host: GPU GTX 1650 4 GB.

**Descartados:** ROS2 Humble (WBC baja a 100-200 Hz), CoppeliaSim (latencia).

---

## Arquitectura

```
Usuario → [PLN] → [Task Planning] → [DRL] → [robot_interface.py] → WBC (Docker)
```

| Módulo | Tecnología | Estado |
|---|---|---|
| PLN | mBERT vs XLM-RoBERTa (fine-tuning, ES+EN) | **En desarrollo (Fase 2)** |
| Task Planning | Planificador jerárquico simple | Por desarrollar |
| DRL | PPO vs SAC (comparar métricas) | Por desarrollar |
| Robot + WBC | ROS Melodic + Gazebo + Docker | Existe y funciona |
| Visión | **Fuera de alcance** — posiciones conocidas + lidar simulado | No se desarrolla |

**Principio:** Arquitectura modular — cada módulo es reemplazable sin afectar los demás.

---

## Robot (caja negra — no modificar WBC)

- Joints base: `mobjoint1/2/3` | Joints brazo: `joint1`–`joint6`
- Spawn: (0,0,0) mirando +X

**Modos WBC** (hardcodeados en `Stack_Tasks.cpp` — pendiente parametrizar en runtime):

| Modo | DOF | Descripción |
|---|---|---|
| `mobile_manipulator` | 9 | Base + brazo coordinados |
| `only_manipulator` | 6 | Solo brazo |
| `mobile_robot` | 3 | Solo base |

**Topics ROS:**

| Topic | Tipo | Dirección |
|---|---|---|
| `/mobile_manipulator/desired_traj` | `Trajectory` | → WBC (comando) |
| `/mobile_manipulator/data` | `MobileManipulator` | ← WBC (estado) |
| `/mobile_manipulator/joint_states` | `JointState` | ← WBC |
| `/mobile_manipulator/commands/velocity` | `Twist` | → base |

**Arm overextension:** Goals dentro de ~1.1 m desde la base → brazo absorbe todo el esfuerzo (pseudoinverso dinámico, menor inercia). **Decisión:** no modificar WBC; se resuelve en Fase 4 con goals siempre fuera del workspace del brazo.

---

## Comunicación

- WebSocket: `roslibpy` + `rosbridge_server` puerto `9090` (integrado en `general_launch.launch`)
- Módulos IA corren **fuera del Docker**; Docker usa `network_mode: host`
- Solo el DRL interactúa con el robot vía WebSocket; PLN y Task Planning se comunican entre sí

**Interfaces:**
```
PLN → Task Planning:   { intent, target, destination, confidence }
Task Planning → DRL:   { task_id, mode, objective{pose}, constraints, subtask_index }
DRL → Robot:           /mobile_manipulator/desired_traj
Robot → DRL:           /mobile_manipulator/data
```

---

## Módulos IA — estructura actual (`modules/`)

```
modules/
  requirements.txt                      ← deps base (roslibpy, numpy)
  robot_interface/
    robot_interface.py                  ← puente roslibpy↔DRL (connect, send_goal, get_state, reset)
  pln/
    requirements_pln.txt                ← transformers, torch, sklearn, pandas
    data/raw/
      seed_dataset.csv                  ← 84 frases semilla (6 intenciones, ES+EN)
      llm_augmentation_prompt.md        ← prompt para aumentar dataset con LLMs
    data/processed/                     ← train/val/test splits (generados por prepare_dataset.py)
    models/                             ← modelos fine-tuneados (gitignored)
    src/
      prepare_dataset.py                ← mezcla, dedup, splits 70/15/15
      dataset.py                        ← PyTorch Dataset
      train.py                          ← fine-tuning --model [mbert|xlmr]
      evaluate.py                       ← métricas comparativas + confusion matrix
      entity_extractor.py               ← extractor de entidades por reglas/regex
      pln_module.py                     ← interfaz final: predict(text) → JSON
```

**PLN — pipeline dos etapas (Opción B):**
1. Clasificación de intención: mBERT o XLM-RoBERTa fine-tuneado (6 clases)
2. Extracción de entidades (target, destination): reglas/regex sobre vocabulario del dominio

**Intenciones:** `navigate`, `pick`, `place`, `fetch`, `transport`, `go_home`

---

## Archivos críticos

| Archivo | Relevancia |
|---|---|
| `CLAUDE.md` | Guía maestra del proyecto |
| `data/data.md` | Jerarquía de la carpeta data/ |
| `data/docker/ros1_docker/Dockerfile` | Imagen Docker |
| `data/docker/ros1_docker/docker-compose.yml` | Config Docker (network: host, GPU) |
| `data/docker/.../general_launch.launch` | Launch principal (rosbridge + arg `gui`) |
| `data/docker/.../Stack_Tasks.cpp` | Modos WBC — **no modificar** |
| `data/docker/.../config.yaml` | Ganancias OSC — **no modificar** |

**Ejecución:**
```bash
roslaunch ... general_launch.launch          # GUI completa (~500 Hz)
roslaunch ... general_launch.launch gui:=false  # headless (~921 Hz, obligatorio para DRL)
```

---

## Estado y fases

- [x] Fase 0: rosbridge, headless, arm overextension analizada
- [x] Fase 1: `robot_interface.py` lista
- [ ] **Fase 2: PLN** — dataset semilla listo, pendiente aumentación LLM + entrenamiento
- [ ] Fase 3: Task Planning
- [ ] Fase 4: DRL (PPO vs SAC, env Gym)
- [ ] Fase 5: Integración end-to-end

**Próximos pasos Fase 2:**
1. Generar `augmented_dataset.csv` con LLMs (ver `pln/data/llm_augmentation_prompt.md`)
2. `python pln/src/prepare_dataset.py`
3. `python pln/src/train.py --model mbert` y `--model xlmr`
4. `python pln/src/evaluate.py`
5. Expandir vocabulario en `entity_extractor.py` con objetos/zonas reales de Gazebo

---

## Decisiones abiertas

- ¿DRL controla solo base (XY) o coordina con brazo?
- ¿Task Planning genera comandos en espacio Cartesiano o de configuración?
- ¿Comparación PPO vs SAC requiere publicación o solo es para la tesis?
- ¿Observación DRL incluye lidar simulado o solo pose del robot?
