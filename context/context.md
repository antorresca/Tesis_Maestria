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
| DRL | DQN principal, PPO como comparación | En desarrollo — módulo creado, primer entrenamiento fallido (ver Fase 4) |
| Robot + WBC | ROS Melodic + Gazebo + Docker | Existe y funciona |
| Visión | **Fuera de alcance** — posiciones conocidas via Gazebo ground truth | No se desarrolla |

**Principio:** Arquitectura modular — cada módulo es reemplazable sin afectar los demás.

---

## Robot (caja negra — no modificar WBC)

- Joints base: `mobjoint1/2/3` | Joints brazo: `joint1`–`joint6`
- Spawn: (0,0,0) mirando +X
- **EE home medido:** `(0.536, 0.000, 0.747)` con todos los joints en 0 (medido 2026-04-02 via `rostopic echo /mobile_manipulator/data -n 1`)

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

**Comportamiento WBC sin goal (descubierto 2026-04-02):** El WBC inicializa `mob_man_traj` (goal interno) a la posición del EE en el DART model cargado desde URDF. Si el WBC arranca antes de que `q_k` tenga valores reales (antes del primer joint_state callback), el DART model puede tener configuración incorrecta y el WBC genera torques explosivos. Fix: `launch-prefix` con `sleep 5` en `osc_controller.launch` via arg `delay`. Sentinel `-10.0` en joints = "mantener posición actual" (confirmado en `updateTarget()`).

**Reset de episodio DRL (mecanismo confirmado 2026-04-03):** `reset_world` sin pausar la física deja al WBC aplicando torques del goal anterior durante el periodo de settle → auto-colisión. Secuencia correcta en `robot_interface.reset()`:
```
pause_physics → reset_world → send_goal(home)×10 → unpause_physics → sleep(settle) → flush_real×10
```
La física se congela mientras el WBC recibe el home goal. Al reanudar, el primer torque ya apunta a home.

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

**Todos los intents pasan por DRL.** El GoalBuilder entrega el `pose_goal` y el DRL lo alcanza sin importar el intent. La diferencia es el Δ que debe cubrir:

| Intent | Δ que el DRL cubre |
|---|---|
| `navigate` | XY + yaw (z de viaje fijo, orientación neutral) |
| `pick` | XY + yaw + Z hacia abajo + orientación top-down |
| `place` | igual que pick, a posición destino |

### Formulación RL

**Observación (9 valores, normalizados):**
```
o = [
  dx_goal / MAX_DIST,   # vector relativo X
  dy_goal / MAX_DIST,   # vector relativo Y
  dz_goal / MAX_DIST,   # vector relativo Z (clave en pick/place)
  dist_goal / MAX_DIST, # distancia euclidiana
  sin(rel_angle),        # ángulo relativo al goal — sin discontinuidad en ±π
  cos(rel_angle),        # ángulo relativo al goal
  obstacle_front,        # 0/1 — sensor geométrico (ground truth Gazebo)
  obstacle_left,         # 0/1
  obstacle_right         # 0/1
]
```

`MAX_DIST = 5.0 m`. `rel_angle = atan2(dy_goal, dx_goal) - yaw_robot`.

**Por qué sin/cos en vez de solo dx/dy:** el DQN no tiene que aprender trigonometría implícita para decidir si girar izquierda o derecha. El ángulo codificado entrega estructura directa al agente.

**Frame stacking (opcional):** apilar N=4 observaciones → 36 inputs. Provee info temporal (velocidad implícita, detección de estancamiento). Implementar con `VecFrameStack` de SB3. Activar si el agente muestra comportamientos oscilatorios sin él.

**Acciones (discretas, 6):**
```
A = { forward, turn_left, turn_right, move_up, move_down, stay }
```

| Acción | Δpose generado |
|---|---|
| `forward` | +STEP_XY en dirección del heading |
| `turn_left` | +TURN_ANGLE en yaw |
| `turn_right` | -TURN_ANGLE en yaw |
| `move_up` | +STEP_Z |
| `move_down` | -STEP_Z |
| `stay` | sin movimiento |

Valores iniciales: `STEP_XY = 0.05 m`, `STEP_Z = 0.03 m`, `TURN_ANGLE = 10°`.

La **orientación del efector** no es una acción: se interpola con slerp (α=0.1) hacia la orientación del goal en cada step. El WBC integra incrementalmente → sin saltos IK.

**Reward:**
```
r  = (dist_prev - dist_curr)   # +si se acerca, -si se aleja
r += -10.0   si colisiona
r += +20.0   si dist < GOAL_THR (éxito)
r += -0.01   penalización base por step
r += -0.05   penalización extra si acción == stay
```

`GOAL_THR = 0.25 m` (amplio para primeros entrenamientos). Colisión geométrica: dist a cualquier obstáculo < `COLLISION_RADIUS = 0.25 m`.

**Sensor de obstáculos — máscara matemática (no LiDAR simulado):**
- Consultar posición de cada obstáculo via `/gazebo/get_model_state`
- Calcular distancia y ángulo relativo al heading del robot
- Asignar sector: front (|rel|<45°), left (45°–135°), right (–135° a –45°)
- Binario: True si dist < `DETECTION_RADIUS = 1.5 m` en ese sector
- Consistente con *omnisciencia de simulación* ya declarada para GoalBuilder

**Algoritmo principal:** DQN (acciones discretas, bajo costo, estable, compatible con frame stacking)
**Comparación académica:** PPO con mismo espacio de obs/acción

**Entorno de entrenamiento:** Gazebo headless (`gui:=false`, ~921 Hz), obstáculos estáticos. Fallback: PyBullet si Gazebo inestable en training loop.

**DEFAULT_TRAINING_GOALS (navigate):** Todos usan orientación identidad (qw=1). Para navigate el WBC mueve la base; el brazo mantiene pose home. Goals con orientaciones grandes (180°) fueron eliminados — causan contorsión del brazo durante el episodio. Goals solo en +X, +Y, −Y y diagonales frente al robot (ninguno detrás).

**WORKSPACE_Z:** `(0.35, 1.2)` para entrenamiento navigate. Evita que exploración aleatoria baje el brazo hasta la base. Ajustar a `(0.10, 1.2)` al pasar a pick/place.

**training_world.world — poses actuales (ajustadas 2026-04-02):**

| Modelo | Pose (x, y, z) | Nota |
|---|---|---|
| `table` | (1.2, 0.0, 0.05) | Plataforma baja 0.8×0.6×0.10 m — base no colisiona lateralmente |
| `red_cube` | (1.2, 0.0, 0.125) | Sobre plataforma; EE baja de z=0.747 → z≈0.15 para pick |
| `tray` | (1.8, -0.8, 0.02) | Destino place |
| `green_zone` | (-0.8, 0.8, 0.01) | Destino navigate |
| `obstacle_1` | (0.7, 0.5, 0.25) | Entre robot y tabla |
| `obstacle_2` | (0.7, -0.5, 0.25) | Entre robot y tabla |

**Baselines de comparación:**
- Baseline 1: Navegación greedy (vector directo al goal, sin evasión)
- Baseline 2 (opcional): A* con mapa conocido de obstáculos

**Métricas de evaluación:**
- Tasa de éxito (% tareas completadas)
- Tiempo promedio (steps hasta goal)
- Número de colisiones
- Eficiencia de trayectoria

**Estructura de archivos DRL:**
```
modules/drl/
  __init__.py
  sensor_utils.py             ← máscara geométrica de obstáculos
  mobile_manipulator_env.py   ← gym.Env sobre robot_interface + sensor
  train.py                    ← entrenamiento DQN y PPO (--algo dqn|ppo)
  evaluate.py                 ← métricas: éxito, steps, colisiones
  config.py                   ← hiperparámetros centralizados
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
| `data/docker/.../general_launch.launch` | Launch para debug con GUI |
| `data/docker/.../drl_launch.launch` | **Entry point DRL** — headless + sin EKF + wbc_delay=5s |
| `data/docker/.../gazebo_only.launch` | Solo Gazebo (paused=true) + rosbridge |
| `data/docker/.../robot_spawn.launch` | Spawn + controladores + WBC (con arg `wbc_delay`) |
| `data/docker/.../osc_controller.launch` | Lanza WBC; arg `delay` agrega sleep antes de arrancar |
| `data/docker/.../init_goal_publisher.py` | Nodo ROS: espera primer data, publica hold-pose 5s |
| `data/docker/.../training_world.world` | Escena DRL con plataforma baja y obstáculos |
| `data/docker/.../Stack_Tasks.cpp` | Modos WBC — **no modificar** |
| `data/docker/.../config.yaml` | Ganancias OSC — **no modificar** |
| `modules/robot_interface/robot_interface.py` | Bridge roslibpy; `reset()` hace pause→reset_world→home×10→unpause→flush_real×10 |
| `modules/drl/config.py` | Hiperparámetros; `_EE_HOME=(0.536,0,0.747)` calibrado; `WORKSPACE_Z=(0.35,1.2)` para navigate |
| `modules/drl/train.py` | Entrenamiento DQN/PPO; incluye `sleep(6s)` de init wait antes de `model.learn()` |

**Ejecución DRL:**
```bash
# Entrenamiento headless (recomendado):
roslaunch mobile_manipulator_unal_description drl_launch.launch world:=training_world

# Debug con GUI:
roslaunch mobile_manipulator_unal_description drl_launch.launch gui:=true world:=training_world

# Desde fuera del Docker:
python -m modules.drl.train --algo dqn --timesteps 500000
```

---

## Estado y fases

- [x] Fase 0: rosbridge, headless, arm overextension analizada
- [x] Fase 1: `robot_interface.py` lista
- [x] **Fase 2: PLN** — XLM-RoBERTa 3 clases atómicas, 94% F1-test, augmentación online ✓
- [x] **Fase 3: Goal Builder** — implementado, probado y validado contra Gazebo ✓
- [ ] Fase 4: DRL (DQN principal, PPO comparación, env Gym sobre Gazebo headless)
  - Módulo creado: `config.py`, `sensor_utils.py`, `mobile_manipulator_env.py`, `train.py`, `evaluate.py` ✓
  - Experimento 01 (500k steps, 16h): no convergió — goals lejanos, Gazebo explotó numéricamente
  - EE home calibrado: `_EE_HOME = (0.536, 0.000, 0.747)` ✓
  - training_world.world ajustado: plataforma baja h=0.10m, objetos más cerca ✓
  - WBC delay fix: `osc_controller.launch` arg `delay`, `drl_launch` pasa `wbc_delay=5` ✓
  - `init_goal_publisher.py` creado: hold-pose al WBC durante 5s al arranque ✓
  - `reset()` corregido: pause→reset_world→home×10→unpause→flush_real×10 ✓
  - Training goals navigate: orientación identidad, sin goals detrás del robot ✓
  - `WORKSPACE_Z = (0.35, 1.2)`: evita brazo bajo durante exploración navigate ✓
  - `train.py`: sleep(6s) antes de model.learn() para esperar init_goal_publisher ✓
  - **Pendiente: smoke test 2000 steps → Entrenamiento 2**
- [ ] Fase 5: Integración end-to-end

**Fase 4 — DRL (fixes completos — pendiente validación smoke test):**
- Tareas al inicio de la próxima sesión:
  1. Lanzar `drl_launch.launch gui:=true world:=training_world`
  2. Confirmar log: `[init_goal_publisher] Hold goal: pos=(0.536, 0.000, 0.747)`
  3. Smoke test: `python -m modules.drl.train --algo dqn --timesteps 2000`
  4. Verificar en Gazebo: resets suaves sin auto-colisión (brazo vuelve a home con pausa física)
  5. Verificar log: `[train] Robot estabilizado — iniciando entrenamiento.`
  6. Si OK → **Entrenamiento 2** completo (`--timesteps 500000`)

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

## Decisiones resueltas (antes abiertas)

| Pregunta | Decisión |
|---|---|
| ¿DRL controla solo base o también efector? | **Todos los DOF necesarios.** Navigate: XY+yaw. Pick/place: XY+yaw+Z+slerp orientación. |
| ¿Entorno de entrenamiento? | **Gazebo headless primero.** Fallback PyBullet solo si hay inestabilidad en training loop. |
| ¿Setup distribuido LAN? | **Mismo PC primero.** Separar solo si hay contención GPU medible. |
| ¿Comparación DQN vs PPO? | Solo para tesis (no publicación). Misma env, mismas métricas, tabla comparativa. |
| ¿Obstáculos estáticos o dinámicos? | **Estáticos** para entrenamiento (training_world.world). Movimiento simple como extensión. |
