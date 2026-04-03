# Experimento 01 — DQN sobre MobileManipulatorEnv (no convergió)

**Fecha:** 2026-03-31 / 2026-04-01  
**Duración aproximada:** ~16 h  
**Algoritmo:** DQN (stable-baselines3)  
**Commit de referencia:** ver `git log modules/drl/`

---

## Configuración

| Parámetro | Valor |
|---|---|
| `TOTAL_TIMESTEPS` | 500 000 |
| `MAX_STEPS` (por episodio) | 500 (al momento del entrenamiento) |
| `GOAL_THR` | 0.25 m |
| `COLLISION_RADIUS` | 0.25 m |
| `WORKSPACE` | X/Y ±2.5 m, Z [0.05, 1.2] m |
| `STEP_XY` | 0.05 m |
| `STEP_Z` | 0.03 m |
| `TURN_ANGLE` | 10° |
| `learning_rate` | 1e-4 |
| `buffer_size` | 100 000 |
| `exploration_fraction` | 0.2 |
| `net_arch` | [128, 128] |
| `eval_freq` | 10 000 steps |
| `n_eval_episodes` | 5 |
| `checkpoint_freq` | 50 000 steps |

**Goals de entrenamiento:** `DEFAULT_TRAINING_GOALS` (6 goals, navigate, Δ ≈ 0.6–0.7 m del EE home placeholder `(0.3, 0.0, 0.4)`)

**Obstáculos:** ninguno (`TRAINING_OBSTACLE_MODELS = []` en ese momento)

---

## Resultados

| Métrica | Valor |
|---|---|
| Evaluaciones totales | 50 (cada 10 k steps) |
| Mean reward en eval #1 (10k steps) | -5.062 |
| Mean reward en eval #50 (500k steps) | -5.147 |
| Mean reward máximo registrado | -4.276 |
| Mean reward mínimo registrado | -5.412 |
| Desviación estándar entre evaluaciones | 0.257 |
| Episodios con éxito (reward > 0) | **0 / 50 evaluaciones** |
| Longitud de episodio (ep_length) | **500 pasos en el 100% de los episodios** |

**Conclusión:** el agente nunca alcanzó ningún goal. La longitud de episodio siempre llegó al máximo (500 steps) y el reward promedio fue plano durante todo el entrenamiento (~-5, sin tendencia apreciable). La mejora total fue de 0.786 reward units entre la primera y la mejor evaluación, sin patrón de convergencia.

---

## Causa raíz identificada

Dos factores críticos impidieron la convergencia:

### 1. Auto-colisión del brazo en Gazebo
Observado con `gui:=true` en sesión posterior al entrenamiento: el brazo del robot colisiona con su propio cuerpo en ausencia de comandos, antes de que el DRL tenga oportunidad de explorar. Esto genera episodios terminados prematuramente por `collision=True` con el reward de `-10`, saturando el replay buffer con experiencias de colisión que el agente no puede aprender a evitar porque ocurren desde el estado inicial.

### 2. EE home sin calibrar (`_EE_HOME` placeholder)
`config.py` tenía `_EE_HOME = (0.3, 0.0, 0.4)` sin medir con `rostopic echo`. Los goals de entrenamiento se generaron como deltas sobre este placeholder, resultando en goals potencialmente fuera del workspace alcanzable del robot o en zonas de auto-colisión.

### 3. Reset con salto del WBC
El mecanismo de reset en `robot_interface.py` usaba `reset_world` antes de enviar el goal home al WBC. Durante ~0.8 s de settle_time, el WBC aplicaba esfuerzos hacia el goal del episodio anterior sobre el robot ya teleportado a origen, generando inestabilidad numérica en Gazebo en algunos episodios.

---

## Archivos de este experimento

| Archivo | Descripción |
|---|---|
| `dqn_model.zip` | Modelo final al step 500 000 |
| `best/dqn/best_model.zip` | Mejor modelo según EvalCallback (step con mean=-4.276) |
| `best/dqn/evaluations.npz` | Historial de 50 evaluaciones: `timesteps`, `results`, `ep_lengths` |
| `logs/dqn/DQN_3/` | TensorBoard logs del run completo (el más largo) |
| `logs/dqn/DQN_1/, DQN_2/` | Runs parciales previos (interrumpidos) |

Los checkpoints intermedios (`checkpoints/dqn/dqn_*_steps.zip`) fueron excluidos de git por `.gitignore` — son equivalentes entre sí (mismo tamaño, misma calidad) y no aportan información adicional al análisis.

---

## Decisiones tomadas a partir de este experimento

1. **Investigar auto-colisión antes del siguiente entrenamiento** (Fase 4, tarea 1).
2. **Medir `_EE_HOME` real** con `rostopic echo /mobile_manipulator/data -n 1` tras un reset limpio.
3. **`drl_launch.launch`** creado para separar Gazebo del robot y reducir carga de tópicos innecesarios (EKF, robot_state_publisher a 1500 Hz).
4. **`StagnationEvalCallback`** implementado en `modules/drl/callbacks.py`: hubiera detenido este experimento en el step ~50 000 (eval #5, mejora < 1.0 en ventana de 5 evals), ahorrando ~15 h de cómputo inútil.
