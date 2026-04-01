# Contexto del Proyecto — Tesis de Profundización

> Leer al inicio de cada sesión. Ver `data/data.md` para jerarquía de archivos Docker/ROS.

## Proyecto

Robot manipulador móvil (base **Robotino3** + brazo **INTERBOTIX VXSA-300**) simulado en **Gazebo/ROS Melodic** dentro de **Docker**. Controlado por **WBC/OSC** a ~921 Hz en simulación. La tesis agrega módulos de IA encima del robot existente. Hardware host: GPU GTX 1650 4 GB.

**Descartados:** ROS2 Humble (WBC baja a 100-200 Hz), CoppeliaSim (latencia).

---

## Arquitectura

```
Usuario → [PLN] → [Goal Builder] → [DRL] → [robot_interface.py] → WBC (Docker)
```

| Módulo | Tecnología | Estado |
|---|---|---|
| PLN | XLM-RoBERTa (`xlm-roberta-base`) — fine-tuning ES+EN | **Completo ✓ (94% F1-test, 3 clases atómicas)** |
| Goal Builder | Mapper semántico JSON→pose + Gazebo ground truth | **Completo ✓** |
| DRL | DQN principal, PPO como comparación | Por desarrollar (Fase 4) |
| Robot + WBC | ROS Melodic + Gazebo + Docker | Existe y funciona |
| Visión | **Fuera de alcance** — posiciones conocidas via Gazebo ground truth | No se desarrolla |

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

**Comportamiento IK del WBC:** Al recibir una pose final lejana, el IK puede resolver con configuraciones discontinuas (codo arriba/abajo). Mitigación: DRL envía incrementos Cartesianos pequeños (Δpose) en cada step, reduciendo ambigüedad IK.

---

## Comunicación

- WebSocket: `roslibpy` + `rosbridge_server` puerto `9090` (integrado en `general_launch.launch`)
- Módulos IA corren **fuera del Docker**; Docker usa `network_mode: host`
- Solo el DRL interactúa con el robot vía WebSocket; PLN y Goal Builder se comunican entre sí

**Interfaces:**
```
PLN → Goal Builder:    { intent, target, destination, confidence }
Goal Builder → DRL:    pose_goal (xyz + orientación) según intent + ground truth Gazebo
DRL → Robot:           /mobile_manipulator/desired_traj  (Δpose incremental por step)
Robot → DRL:           /mobile_manipulator/data
```

**Opción de setup distribuido (si hay contención GPU):**
- Gazebo headless corre en PC secundaria (GTX 1350) — `robot_interface.py` apunta a IP de esa máquina en lugar de `localhost`
- Entrenamiento DRL en PC principal (GTX 1650)
- Cambio de código mínimo: solo la IP en `robot_interface.py`
- **Recomendación:** probar primero en mismo PC (Gazebo headless es mayormente CPU); pasar a LAN solo si hay contención real

---

## Módulo Goal Builder

**Responsabilidad:** Recibe el JSON del PLN, consulta la posición del objeto/destino vía Gazebo ground truth (`/gazebo/get_model_state`), y construye la `pose_goal` completa que recibe el DRL.

**Supuesto clave (declarado en tesis):** Posiciones de objetos conocidas globalmente (omnisciencia de simulación). Justificado porque el foco de la tesis es la arquitectura PLN→DRL→WBC, no percepción.

**Diferencia de pose_goal por intent:**

| Intent | x, y | z | Orientación efector |
|---|---|---|---|
| `navigate` | posición frente al objetivo | z "seguro" (altura de viaje) | estable, mirando al objetivo |
| `pick` | posición del objeto | z de agarre | normal a la superficie del objeto |
| `place` | posición destino | z de depósito | normal a la superficie destino |

Pick y place son operacionalmente el mismo tipo (bajar efector con orientación específica), pero a localizaciones distintas. Se mantienen como clases separadas por claridad semántica.

---

## Módulo DRL — Fase 4

**Rol:** Planificador de trayectoria reactivo en espacio Cartesiano. Aprende a llegar al `pose_goal` enviando Δpose incrementales al WBC, evitando obstáculos locales bajo observabilidad parcial (POMDP simplificado).

**No controla:** planificación global, percepción, lógica de pick/place (eso lo determina el Goal Builder).

### Formulación RL

**Observación:**
```
o = [
  dx_goal, dy_goal, dz_goal,   # vector relativo al objetivo (Cartesiano)
  dist_goal,                    # distancia euclidiana al objetivo
  obstacle_front,               # 0/1 (lidar simulado)
  obstacle_left,                # 0/1
  obstacle_right                # 0/1
]
```

**Acciones (discretas):**
```
A = { forward, turn_left, turn_right, stay }
```
Cada acción genera un Δpose pequeño que se envía al WBC vía `/mobile_manipulator/desired_traj`.

**Reward:**
```
+1    si reduce distancia al goal
-1    si aumenta distancia
-10   si colisiona
+20   si alcanza el objetivo (dist < umbral)
-0.01 penalización por step (eficiencia)
```

**Algoritmo principal:** DQN (acciones discretas, bajo costo computacional, estable)
**Comparación académica:** PPO con mismo espacio de observación/acción

**Entorno de entrenamiento:** Gazebo headless (`gui:=false`, ~921 Hz) con 2–5 obstáculos (estáticos o con movimiento simple). Fallback: PyBullet si Gazebo presenta inestabilidad en el loop de entrenamiento.

**Baselines de comparación:**
- Baseline 1: Navegación greedy (ir directo al objetivo sin evitar obstáculos)
- Baseline 2 (opcional): A* con conocimiento incompleto

**Métricas de evaluación:**
- Tasa de éxito (% tareas completadas)
- Tiempo promedio (steps hasta goal)
- Número de colisiones
- Eficiencia de trayectoria

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
1. Clasificación de intención: **XLM-RoBERTa fine-tuneado** (94% F1, 6 ms/sample, 1112 MB)
2. Extracción de entidades (target, destination): reglas/regex sobre vocabulario del dominio

**Intenciones (3 clases atómicas):** `navigate`, `pick`, `place`

> go_home → navigate (destination="home"), fetch → pick, transport → place. Goal Builder descompone en pose_goal con parámetros distintos por intent.

**Config de entrenamiento:** `--unfreeze_layers 0` (full fine-tuning), `max_length=64`, `batch_size=4`, `grad_accum=4`, `weight_decay=0.1`, early stopping `patience=3` sobre `eval_val_f1_weighted`

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
- [x] **Fase 2: PLN** — XLM-RoBERTa 3 clases atómicas, 94% F1-test, augmentación online ✓
- [x] **Fase 3: Goal Builder** — implementado, probado y validado contra Gazebo ✓
- [ ] Fase 4: DRL (DQN principal, PPO comparación, env Gym sobre Gazebo headless)
- [ ] Fase 5: Integración end-to-end

**Fase 3 — Goal Builder (en progreso):**
- `entity_extractor.py` y `pln_module.py` ya soportan 3 clases + XLM-R ✓
- `modules/goal_builder/goal_builder.py` implementado:
  - Recibe JSON PLN, consulta `/gazebo/get_model_state`, retorna `pose_goal`
  - Lookup table `ENTITY_TO_GAZEBO` en el mismo archivo (expandir por escena)
  - Compatible con `robot_interface.send_goal()` (mismo formato xyz + cuaternión)
  - Acepta `ros_client` externo para compartir conexión con DRL

---

## Limitaciones conocidas (declarar en tesis)

| Limitación | Módulo | Alcance |
|---|---|---|
| Unicidad semántica de objetos | Goal Builder | La escena asume un objeto por tipo semántico. Dos cubos rojos son indistinguibles para el sistema sin contexto espacial o percepción. |
| Vocabulario cerrado | entity_extractor | Solo reconoce entidades del dominio definidas en `OBJECTS`/`LOCATIONS`. Entidades fuera del vocabulario generan `[WARN]` sin romper el pipeline. |
| Posiciones estáticas | Goal Builder | Las posiciones se consultan en el momento de `build()`. Si un objeto se mueve después (e.g., tras un pick), el Goal Builder no lo sabe hasta la siguiente consulta. |

## Trabajo futuro

- **World Describer Agent:** Un agente que consulte `/gazebo/get_world_properties` + `/gazebo/get_model_state` y genere descripciones en lenguaje natural del estado del mundo (`"hay un cubo rojo en la mesa a 2m"`). Encajaría como módulo entre Gazebo y el PLN/GoalBuilder, cerrando el loop de grounding sin modificar ningún módulo existente. Arquitectura modular lo permite directamente.
- **Módulo de visión:** Reemplazar Gazebo ground truth por detección visual real (YOLO + estimación de pose).
- **HRL sobre Goal Builder:** Añadir un planificador jerárquico que descomponga tareas compuestas ("lleva el cubo rojo a la bandeja desde la cocina") en secuencias de intents atómicos.

---

## Decisiones abiertas

- ¿DRL controla solo base (XY+heading) o también Δpose del efector final para pick/place?
- ¿Entorno de entrenamiento: Gazebo headless directo o PyBullet para velocidad?
- ¿Setup distribuido LAN: Gazebo en PC2 (GTX 1350), entrenamiento en PC1 (GTX 1650)?
- ¿Comparación DQN vs PPO requiere publicación o solo es para la tesis?
- ¿Obstáculos del entorno: estáticos, semi-dinámicos, o ambos?
