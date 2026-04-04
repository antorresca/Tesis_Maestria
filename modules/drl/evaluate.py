"""
evaluate.py — Evaluación de un modelo DRL entrenado (PyBullet).

Métricas reportadas por episodio y en promedio:
    - Tasa de éxito          (% episodios que alcanzan el goal)
    - Steps promedio         (solo episodios exitosos)
    - Colisiones             (total y por episodio)
    - Eficiencia de trayecto (distancia recta / distancia recorrida)

Uso:
    # Evaluar mejor modelo DQN con GUI
    python -m modules.drl.evaluate --model models/drl/best/dqn/best_model --gui

    # Sin GUI, más episodios
    python -m modules.drl.evaluate --model models/drl/dqn_model --episodes 20

    # Baseline greedy (sin DRL)
    python -m modules.drl.evaluate --baseline greedy --episodes 20
"""

import argparse
import math
import os

import numpy as np

try:
    from .mobile_manipulator_env import MobileManipulatorEnv
    from .config import (ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
                         GOAL_THR, TRAINING_OBSTACLE_MODELS)
except ImportError:
    from mobile_manipulator_env import MobileManipulatorEnv
    from config import (ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
                        GOAL_THR, TRAINING_OBSTACLE_MODELS)


# ---------------------------------------------------------------------------
# Políticas baseline
# ---------------------------------------------------------------------------

def _greedy_action(obs: np.ndarray) -> int:
    """
    Baseline greedy: avanza hacia el goal girando si el ángulo relativo
    es suficientemente grande. Sin evasión de obstáculos.

    obs[4] = sin(rel_angle), obs[5] = cos(rel_angle)
    """
    sin_a = obs[4]
    cos_a = obs[5]
    rel_a = math.atan2(sin_a, cos_a)   # ángulo relativo al goal (rad)

    TURN_THR = math.radians(15.0)
    if rel_a > TURN_THR:
        return ACTION_TURN_LEFT
    elif rel_a < -TURN_THR:
        return ACTION_TURN_RIGHT
    else:
        return ACTION_FORWARD


# ---------------------------------------------------------------------------
# Función de evaluación
# ---------------------------------------------------------------------------

def evaluate(
    env: MobileManipulatorEnv,
    policy_fn,
    n_episodes: int = 20,
    verbose: bool = True,
    seed: int = 42,
) -> dict:
    """
    Ejecuta n_episodes episodios con policy_fn y recopila métricas.

    policy_fn : callable(obs) → action_int

    Returns dict con:
        success_rate        float [0,1]
        avg_steps           float (promedio sobre episodios exitosos)
        total_collisions    int
        collision_rate      float [0,1]
        avg_path_efficiency float (straight / actual, solo ep. exitosos)
        per_episode         list de dicts
    """
    np.random.seed(seed)

    successes   = 0
    collisions  = 0
    steps_list  = []        # steps de episodios exitosos
    efficiency_list = []    # eficiencias de episodios exitosos
    per_episode = []

    for ep in range(n_episodes):
        obs, info = env.reset()

        # Posición inicial del efector para calcular path_length
        init_state  = env._ri.get_state()
        init_pos    = init_state["ee_pos"] if init_state else [0.0, 0.0, 0.0]
        goal        = env._pose_goal
        straight_d  = math.sqrt(
            (goal["x"] - init_pos[0])**2 +
            (goal["y"] - init_pos[1])**2 +
            (goal["z"] - init_pos[2])**2
        )

        ep_steps      = 0
        ep_collisions = 0
        ep_success    = False
        path_length   = 0.0
        prev_pos      = init_pos[:]

        terminated = truncated = False
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)

            ep_steps += 1
            if step_info.get("collision", False):
                ep_collisions += 1

            # Acumular longitud de trayectoria
            cur_state = env._ri.get_state()
            if cur_state:
                cp = cur_state["ee_pos"]
                path_length += math.sqrt(
                    (cp[0] - prev_pos[0])**2 +
                    (cp[1] - prev_pos[1])**2 +
                    (cp[2] - prev_pos[2])**2
                )
                prev_pos = cp[:]

            if step_info.get("success", False):
                ep_success = True

        successes  += int(ep_success)
        collisions += ep_collisions

        ep_efficiency = (straight_d / path_length) if (ep_success and path_length > 1e-6) else 0.0
        if ep_success:
            steps_list.append(ep_steps)
            efficiency_list.append(ep_efficiency)

        ep_result = {
            "episode":    ep + 1,
            "success":    ep_success,
            "steps":      ep_steps,
            "collisions": ep_collisions,
            "efficiency": ep_efficiency,
        }
        per_episode.append(ep_result)

        if verbose:
            tag = "✓" if ep_success else "✗"
            print(f"  [{tag}] Ep {ep+1:3d} | steps={ep_steps:4d} | "
                  f"cols={ep_collisions} | eff={ep_efficiency:.3f}")

    success_rate    = successes / n_episodes
    avg_steps       = float(np.mean(steps_list))       if steps_list      else 0.0
    collision_rate  = collisions / n_episodes
    avg_efficiency  = float(np.mean(efficiency_list))  if efficiency_list else 0.0

    return {
        "success_rate":        success_rate,
        "avg_steps":           avg_steps,
        "total_collisions":    collisions,
        "collision_rate":      collision_rate,
        "avg_path_efficiency": avg_efficiency,
        "per_episode":         per_episode,
    }


def print_summary(label: str, metrics: dict, n_episodes: int):
    print(f"\n{'='*50}")
    print(f"  {label}  ({n_episodes} episodios)")
    print(f"{'='*50}")
    print(f"  Tasa de éxito          : {metrics['success_rate']*100:.1f}%")
    print(f"  Steps promedio (éxito) : {metrics['avg_steps']:.1f}")
    print(f"  Colisiones totales     : {metrics['total_collisions']}")
    print(f"  Tasa de colisión       : {metrics['collision_rate']*100:.1f}%")
    print(f"  Eficiencia promedio    : {metrics['avg_path_efficiency']:.3f}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluación DRL — robot manipulador móvil (PyBullet)")
    p.add_argument("--model",     default=None,
                   help="Ruta al modelo .zip (SB3). Requerido si --baseline no se usa.")
    p.add_argument("--algo",      choices=["dqn", "ppo"], default="dqn",
                   help="Algoritmo del modelo cargado (default: dqn)")
    p.add_argument("--baseline",  choices=["greedy"], default=None,
                   help="Evaluar política baseline en lugar del modelo DRL")
    p.add_argument("--episodes", "--n_episodes", dest="n_episodes",
                   type=int, default=20,
                   help="Número de episodios de evaluación (default: 20)")
    p.add_argument("--gui",       action="store_true",
                   help="Abrir ventana gráfica de PyBullet")
    p.add_argument("--obstacle_models", nargs="*", default=None,
                   help="Obstáculos a incluir (default: TRAINING_OBSTACLE_MODELS)")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla para reproducibilidad (default: 42) — usar el mismo valor al comparar DQN vs PPO")
    return p.parse_args()


def main():
    args = parse_args()

    if args.baseline is None and args.model is None:
        print("[evaluate] Error: se debe indicar --model o --baseline.")
        return

    obstacle_models = args.obstacle_models if args.obstacle_models is not None \
                      else TRAINING_OBSTACLE_MODELS

    env = MobileManipulatorEnv(
        gui=args.gui,
        obstacle_models=obstacle_models,
    )

    # ---- Política -------------------------------------------------------
    if args.baseline == "greedy":
        label     = "Baseline Greedy"
        policy_fn = _greedy_action
    else:
        from stable_baselines3 import DQN, PPO
        _cls = {"dqn": DQN, "ppo": PPO}[args.algo]
        model = _cls.load(args.model, env=env)
        label = f"{args.algo.upper()} ({os.path.basename(args.model)})"
        policy_fn = lambda obs: int(model.predict(obs, deterministic=True)[0])

    # ---- Evaluación -----------------------------------------------------
    print(f"\n[evaluate] {label} — {args.n_episodes} episodios  (seed={args.seed})")
    metrics = evaluate(env, policy_fn, n_episodes=args.n_episodes, verbose=True, seed=args.seed)
    print_summary(label, metrics, args.n_episodes)

    env.close()


if __name__ == "__main__":
    main()
