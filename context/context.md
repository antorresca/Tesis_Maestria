# Contexto del Proyecto — Tesis de Profundización

> Leer al inicio de cada sesión.

## Proyecto

Robot manipulador móvil (base **Robotino3** + brazo **INTERBOTIX VXSA-300**) simulado en **PyBullet** (migración desde Gazebo/ROS Melodic). El WBC/OSC de Gazebo se reemplazó con control cinemático directo en PyBullet para el entrenamiento DRL. La tesis agrega módulos de IA sobre la plataforma robótica. Hardware host: GPU GTX 1650 4 GB.

**URDF fuente:** `data_legacy/src/` — paquetes ROS1 conservados solo para referencia del URDF. URDF compilado para PyBullet: `modules/robot_interface/urdf/mobile_manipulator_pybullet.urdf`.

**Descartados:** Gazebo/ROS Melodic+Docker (explosiones numéricas en training loop DRL), ROS2 Humble (WBC baja a 100-200 Hz), CoppeliaSim (latencia).

---

## Arquitectura

```
Usuario → [PLN] → [Goal Builder] → [DRL Agente 1 (XY)] → [DRL Agente 2 (Z)] → [PyBulletRobot]
```

| Módulo | Tecnología | Estado |
|---|---|---|
| PLN | XLM-RoBERTa (`xlm-roberta-base`) — fine-tuning ES+EN | **Completo ✓ (94% F1-test, 3 clases atómicas)** |
| Goal Builder | Mapper semántico JSON→pose + PyBullet ground truth | **Completo ✓** |
| DRL Agente 1 | DQN/PPO (disc) + SAC/PPO (cont) — navegación XY | **Completo ✓** — DQN 86% best; ver tabla bitácora |
| DRL Agente 2 | DQN + PPO — posicionamiento Z (brazo) | **Completo ✓** — DQN 85%, PPO 87%, baseline 87% (todos equivalentes) |
| Robot sim | **PyBullet** — control cinemático + URDF del Robotino3+VXSA300 | Migrado ✓ (Gazebo/Docker eliminados) |
| Visión | **Fuera de alcance** — posiciones conocidas via PyBullet ground truth | No se desarrolla |

**Principio:** Arquitectura modular — cada módulo es reemplazable sin afectar los demás.

---

## Robot (PyBullet — control cinemático)

- Joints brazo (URDF): `waist_joint`, `shoulder_joint`, `elbow_joint`, `upper_forearm_joint`, `lower_forearm_joint`, `wrist_joint`
- EE link: `mobile_manipulator/end_effector_link`
- Spawn: (0,0,0) mirando +X
- **EE home medido (PyBullet FK, joints=0):** `(0.472, 0.000, 0.747)` — medido con `test_pybullet_setup.py`. En Gazebo era 0.536 (diferencia de frame DART vs PyBullet, ~6 cm en X).

**Control en PyBulletRobot (`pybullet_robot.py`):**
- `send_goal(x, y, z, qx, qy, qz, qw) → bool`:
  1. Guarda pose anterior (base + joints) para poder revertir
  2. Extrae yaw del cuaternión
  3. Posiciona base para que EE quede en (x,y): `base_xy = ee_xy - R(yaw) @ (0.472, 0)`
  4. `resetBasePositionAndOrientation` (kinematic, sin física)
  5. IK PyBullet → `resetJointState` en los 6 joints del brazo
  6. `performCollisionDetection` + `getContactPoints` — si hay contacto con obstáculo registrado → revierte pose, retorna `True`
- `reset()`: `resetBasePositionAndOrientation([0,0,0])` + `resetJointState(0)` en todos + `send_goal(home)`
- `get_state()`: `getLinkState(ee_link, computeForwardKinematics=True)` + `getJointState` por joint

**Colisión física:**
- `send_goal` retorna `True` si el movimiento fue bloqueado por contacto real con obstáculo
- El filtro es `bodyB in obstacle_body_ids` — excluye plano y auto-colisión del URDF
- `mobile_manipulator_env.step()` usa este bool como señal de colisión (no el check geométrico)

**Obstáculos en PyBullet:**
- `add_obstacle(model_name, position, half_extents)` crea cuerpos estáticos con visual rojo
- `get_obstacle_position(model_name)` reemplaza `/gazebo/get_model_state`
- Specs definidas en `OBSTACLE_SPECS` de `config.py`

---

## Comunicación

- Todo corre en el **mismo proceso Python** — no más WebSocket/ROS
- Solo el DRL interactúa con la simulación vía `PyBulletRobot`; PLN y Goal Builder se comunican entre sí

**Interfaces:**
```
PLN → Goal Builder:    { intent, target, destination, confidence }
Goal Builder → DRL:    pose_goal (xyz + orientación) según intent + ground truth PyBullet
DRL Agente 1 → Robot:  PyBulletRobot.send_goal(x,y,z_home,qx,qy,qz,qw)  — navega XY
DRL Agente 2 → Robot:  PyBulletRobot.send_goal(x_curr,y_curr,z_goal,...) — ajusta Z
Robot → DRL:           PyBulletRobot.get_state() → {ee_pos, ee_rpy, joints}
```

---

## Módulo Goal Builder

**Responsabilidad:** Recibe el JSON del PLN, consulta la posición del objeto/destino vía PyBullet ground truth (`PyBulletRobot.get_obstacle_position()`), y construye la `pose_goal` completa que recibe el DRL.

**Supuesto clave (declarado en tesis):** Posiciones de objetos conocidas globalmente (omnisciencia de simulación). Justificado porque el foco de la tesis es la arquitectura PLN→DRL→WBC, no percepción.

**Diferencia de pose_goal por intent:**

| Intent | x, y | z | Orientación efector |
|---|---|---|---|
| `navigate` | posición frente al objetivo | z "seguro" (altura de viaje = _EE_HOME[2]) | estable, mirando al objetivo |
| `pick` | posición del objeto | z de agarre | normal a la superficie del objeto |
| `place` | posición destino | z de depósito | normal a la superficie destino |

Pick y place son operacionalmente el mismo tipo (bajar efector con orientación específica), pero a localizaciones distintas. Se mantienen como clases separadas por claridad semántica.

---

## Módulo DRL — Fase 4

**Diseño de dos agentes (2026-04-04):**

El DRL se divide en dos agentes especializados en lugar de un agente 3D unificado. La razón: navigate solo requiere XY, pick/place activan ambos. Mezclar las señales de aprendizaje degrada la convergencia del agente XY.

| | Agente 1 — Navegación | Agente 2 — Posicionamiento |
|---|---|---|
| **Controla** | Base (x, y, heading) | Brazo (z) |
| **Acciones** | FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, STAY | MOVE_UP, MOVE_DOWN, STAY |
| **Intent activo** | Siempre | Solo pick/place |
| **Estado** | En entrenamiento (Entrenamiento 9) | Pendiente |

**Flujo por intent:**
```
navigate  → Agente 1 lleva base a (x, y)      — Agente 2 inactivo (z=home)
pick      → Agente 1 lleva base a (x, y)      → Agente 2 baja brazo a z_goal
place     → Agente 1 lleva base a (x, y)      → Agente 2 sube brazo a z_goal
```

### Formulación RL — Agente 1

**Observación (9 valores, normalizados):**
```
o = [
  dx_goal / MAX_DIST,    # vector relativo X
  dy_goal / MAX_DIST,    # vector relativo Y
  dz_goal / MAX_DIST,    # vector relativo Z (siempre ≈0 para Agente 1)
  dist_goal / MAX_DIST,  # distancia euclidiana
  sin(rel_angle),         # ángulo entre cmd_heading y goal — sin discontinuidad en ±π
  cos(rel_angle),         # ángulo entre cmd_heading y goal
  obstacle_front,         # proximidad normalizada [0,1] — 1.0 = a COLLISION_RADIUS
  obstacle_left,          # proximidad normalizada [0,1]
  obstacle_right          # proximidad normalizada [0,1]
]
```

`rel_angle = atan2(dy_goal, dx_goal) - self._cmd_heading` (**crítico**: usar `cmd_heading`, no `ee_rpy[2]`).

Obs[6/7/8] son proximidad **continua** (no binaria): `proximity = (DETECTION_RADIUS - dist) / (DETECTION_RADIUS - COLLISION_RADIUS)`, tomando el máximo por sector.

**Acciones (discretas, 5) — Agente 1:**
```
A = { forward, turn_left, turn_right, backward, stay }
```

| Acción | Δpose generado |
|---|---|
| `forward` | +STEP_XY en dirección del cmd_heading |
| `backward` | -STEP_XY en dirección del cmd_heading |
| `turn_left` | +TURN_ANGLE en cmd_heading |
| `turn_right` | -TURN_ANGLE en cmd_heading |
| `stay` | sin movimiento |

`STEP_XY = 0.05 m`, `TURN_ANGLE = 10°`.

**Reward — Agente 1:**
```
r  = (dist_prev - dist_curr)                              # shaping: + si se acerca
r += -20.0   si colisión física (send_goal retorna True)
r += +20.0   si dist < GOAL_THR (éxito)
r += -0.01   penalización base por step
r += -0.05   penalización extra si acción == stay
r += -REWARD_PROXIMITY_SCALE * (front + 0.5*max(left,right))  # continua [0,1]
```

`GOAL_THR = 0.25 m`. `COLLISION_RADIUS = 0.25 m`. `DETECTION_RADIUS = 1.5 m`. `REWARD_PROXIMITY_SCALE = 0.10`.

**Sensor:** `SENSOR_OBSTACLE_MODELS = ['obstacle_1', 'obstacle_2']` — la tabla se excluye del sensor (es destino de pick/place, no peligro de navegación). La colisión física con la tabla sigue activa.

**Algoritmos:** DQN y PPO — misma env, mismo seed=42 para evaluación.

**Goals de entrenamiento — Agente 1 (muestreo aleatorio):**
- En cada episodio se muestrea (x, y) uniformemente de `GOAL_SAMPLE_X=(-1.5, 2.3)` × `GOAL_SAMPLE_Y=(-1.5, 1.5)`
- Se rechazan goals dentro de obstáculos (margen COLLISION_RADIUS+0.05) o a menos de 0.5 m del spawn
- Fuerza generalización: el agente no puede memorizar trayectorias fijas
- `DEFAULT_TRAINING_GOALS` (17 goals) se mantiene como fallback y para evaluación con seed fijo

**Obstáculos training (OBSTACLE_SPECS en config.py):**

| Modelo | Posición | Half extents |
|---|---|---|
| `obstacle_1` | (0.7, +0.7, 0.25) | (0.15, 0.15, 0.25) |
| `obstacle_2` | (0.7, −0.7, 0.25) | (0.15, 0.15, 0.25) |
| `table` | (2.0, 0.0, 0.05) | (0.40, 0.30, 0.05) |

Pasillo libre entre obstacle_1 y obstacle_2: 1.1m (y: -0.55 a +0.55 m libre).

**Hiperparámetros DQN actuales (config.py):**
```python
DQN_HPARAMS = {
    "learning_rate":           1e-4,
    "buffer_size":             200_000,
    "learning_starts":         2_000,
    "batch_size":              128,
    "tau":                     1.0,
    "gamma":                   0.99,
    "train_freq":              4,
    "gradient_steps":          1,
    "target_update_interval":  1_000,
    "exploration_fraction":    0.15,   # explora hasta step 150k de 1M
    "exploration_final_eps":   0.10,
    "policy_kwargs":           {"net_arch": [256, 128]},
}
TOTAL_TIMESTEPS = 1_000_000
MAX_STEPS = 500
```

**Ejecución DRL:**
```bash
# Entrenamiento DQN:
python -m modules.drl.train --algo dqn --timesteps 1000000 --patience 50

# Entrenamiento PPO:
python -m modules.drl.train --algo ppo --timesteps 1000000 --patience 75

# Evaluación DQN:
python -m modules.drl.evaluate --model models/drl/best/dqn/best_model --algo dqn --episodes 100 --seed 42

# Evaluación PPO:
python -m modules.drl.evaluate --model models/drl/best/ppo/best_model --algo ppo --episodes 100 --seed 42

# Demo visual:
python -m modules.drl.demo --model models/drl/best/dqn/best_model --all_goals --delay 0.2
python -m modules.drl.demo --model models/drl/best/dqn/best_model --x 1.5 --y 0.5 --delay 0.3
```

---

## Módulos IA — estructura actual (`modules/`)

```
modules/
  requirements.txt
  robot_interface/
    pybullet_robot.py             ← simulador cinemático PyBullet (reemplaza robot_interface.py)
    urdf/
      mobile_manipulator_pybullet.urdf  ← URDF limpio (sin transmissions/gazebo)
  drl/
    config.py                     ← hiperparámetros centralizados (Agente 1 + constantes compartidas)
    mobile_manipulator_env.py     ← Gym.Env Agente 1 (XY, Discrete(5))
    pybullet_sensor_utils.py      ← proximidad continua por sectores angulares
    train.py                      ← entrenamiento DQN/PPO con --gui, --patience
    evaluate.py                   ← métricas: éxito, steps, colisiones, eficiencia
    demo.py                       ← visualización interactiva con delay configurable
    callbacks.py                  ← StagnationEvalCallback (early stop por patience)
  drl_agent2/
    __init__.py
    config.py                     ← hiperparámetros específicos del brazo (ARM_*)
    arm_env.py                    ← Gym.Env Agente 2 (Z, Discrete(3), obs=3)
    train.py                      ← entrenamiento DQN/PPO Agente 2
    evaluate.py                   ← métricas: éxito, error Z, eficiencia
    demo.py                       ← demo visual con z_goal configurable
  pln/
    ...                           ← XLM-RoBERTa fine-tuneado (completo ✓)
  goal_builder/
    goal_builder.py               ← completo ✓
```

**PLN — pipeline dos etapas:**
1. Clasificación de intención: **XLM-RoBERTa fine-tuneado** (94% F1, 6 ms/sample)
2. Extracción de entidades: reglas/regex sobre vocabulario del dominio

**Intenciones (3 clases atómicas):** `navigate`, `pick`, `place`

---

## Archivos críticos

| Archivo | Relevancia |
|---|---|
| `CLAUDE.md` | Guía maestra del proyecto |
| `context/context.md` | Este archivo |
| `data_legacy/src/` | Paquetes ROS1 — solo referencia URDF |
| `modules/robot_interface/urdf/mobile_manipulator_pybullet.urdf` | URDF limpio |
| `modules/robot_interface/pybullet_robot.py` | Simulador PyBullet con colisión física |
| `modules/drl/pybullet_sensor_utils.py` | Proximidad continua por sectores |
| `modules/drl/config.py` | `_EE_HOME=(0.472,0,0.747)`; acciones Agente 1; `OBSTACLE_SPECS` |
| `modules/drl/mobile_manipulator_env.py` | Gym.Env Agente 1 — Discrete(5) |
| `modules/drl/train.py` | Entrenamiento DQN/PPO |
| `modules/drl/demo.py` | Visualización paso a paso con delay |
| `models/drl/best/dqn/best_model` | Modelo DQN Agente 1 — Entrenamiento 13, goals aleatorios, **86% éxito** |
| `models/drl/best/ppo/best_model` | Modelo PPO Agente 1 — PPO-2 en entrenamiento (PPO-1: 62% éxito) |
| `modules/drl_agent2/arm_env.py` | Gym.Env Agente 2 — Discrete(3), obs=[dz_norm, z_norm, dist_z_norm] |
| `modules/drl_agent2/config.py` | Hiperparámetros Agente 2: `GOAL_Z_THR=0.05m`, `ARM_MAX_STEPS=150`, `ARM_TOTAL_TIMESTEPS=300k` |
| `modules/drl/continuous_env.py` | `ContinuousManipulatorEnv` — Box(2) `[v_linear, v_angular]`, misma obs/reward que discreto |

---

## Estado y fases

- [x] Fase 0: rosbridge, headless, arm overextension analizada
- [x] Fase 1: `robot_interface.py` lista
- [x] **Fase 2: PLN** — XLM-RoBERTa 3 clases atómicas, 94% F1-test ✓
- [x] **Fase 3: Goal Builder** — implementado y validado ✓
- [ ] **Fase 4: DRL**
  - [x] Migración PyBullet completa (2026-04-03) ✓
  - [x] Colisión física real implementada (2026-04-04) ✓
  - [x] Proximidad continua en sensor (2026-04-04) ✓
  - [x] Arquitectura dos agentes definida (2026-04-04) ✓
  - [x] Muestreo aleatorio de goals + sensor sin tabla ✓
  - [x] **DQN Agente 1:** 86% éxito, 14% col, 0% timeout ✓
  - [x] **Comparación DQN vs PPO completada** ✓ — DQN supera PPO (86% vs 65%); ver bitácora
  - [x] **Agente 2 (Z) — completo** ✓ — DQN 85% / PPO 87% / baseline proporcional 87% (todos equivalentes; se usará baseline en producción)
  - [x] **Agente 1 continuo — completo** ✓ — SAC 66% / PPO-cont 76% / DQN discreto sigue siendo el mejor (86%)
  - [ ] **Fase 5: integración end-to-end**
- [ ] Fase 5: Integración end-to-end

---

## Limitaciones conocidas (declarar en tesis)

| Limitación | Módulo | Alcance |
|---|---|---|
| Unicidad semántica de objetos | Goal Builder | La escena asume un objeto por tipo semántico |
| Vocabulario cerrado | entity_extractor | Solo entidades del dominio definidas |
| Posiciones estáticas | Goal Builder | Las posiciones se consultan en el momento de `build()` |
| Control cinemático (no dinámico) | PyBulletRobot | El robot teleporta — no simula inercia ni fricción real |
| Goals fuera del workspace | Agente 1 | Goals con x>2.5 o fuera de WORKSPACE son imposibles |

## Trabajo futuro

- ~~**Agente 2 (Z):** `ArmEnv` minimalista — obs=[dz, dist_z], acciones=3, sin obstáculos~~ **Implementado** ✓ en `modules/drl_agent2/`
- **World Describer Agent:** consulta PyBullet ground truth y genera descripciones en lenguaje natural
- **Módulo de visión:** Reemplazar ground truth por detección visual real (YOLO + estimación de pose)
- **HRL sobre Goal Builder:** Planificador jerárquico para tareas compuestas

---

## Decisiones resueltas

| Pregunta | Decisión |
|---|---|
| ¿DRL controla solo base o también efector? | **Dos agentes:** Agente 1 (XY, base), Agente 2 (Z, brazo). Navigate solo usa Agente 1. |
| ¿Entorno de entrenamiento? | **PyBullet** — Gazebo/Docker eliminados. ~251 it/s headless. |
| ¿Comparación DQN vs PPO? | Solo para tesis. Misma env, mismas métricas, tabla comparativa. |
| ¿Obstáculos estáticos o dinámicos? | **Estáticos** para entrenamiento. |
| ¿Colisión geométrica o física? | **Física** — `performCollisionDetection` + `getContactPoints` contra `obstacle_body_ids` |
| ¿Obs de obstáculos binaria o continua? | **Continua** [0,1] — permite graduar respuesta antes de la colisión |
