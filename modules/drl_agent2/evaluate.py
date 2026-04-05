"""
evaluate.py — Evaluación de un modelo Agente 2 entrenado (posicionamiento Z).

Métricas reportadas por episodio y en promedio:
    - Tasa de éxito     (% episodios con |dz| < GOAL_Z_THR)
    - Steps promedio    (solo episodios exitosos)
    - Error Z final     (|dz| al final del episodio)
    - Eficiencia        (steps mínimos teóricos / steps reales, solo exitosos)

Uso:
    # DQN
    python -m modules.drl_agent2.evaluate --model models/drl_agent2/best/dqn/best_model --algo dqn

    # PPO
    python -m modules.drl_agent2.evaluate --model models/drl_agent2/best/ppo/best_model --algo ppo

    # Baseline proporcional (sube/baja según signo de dz)
    python -m modules.drl_agent2.evaluate --baseline proportional --episodes 50

    # Con GUI
    python -m modules.drl_agent2.evaluate --model models/drl_agent2/best/dqn/best_model --gui
"""

import argparse
import math
import os

import numpy as np

try:
    from .arm_env import ArmEnv
    from .config import GOAL_Z_THR
    from ..drl.config import ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_ARM_STAY, STEP_Z
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from drl_agent2.arm_env import ArmEnv
    from drl_agent2.config import GOAL_Z_THR
    from drl.config import ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_ARM_STAY, STEP_Z


# ---------------------------------------------------------------------------
# Políticas baseline
# ---------------------------------------------------------------------------

def _proportional_action(obs: np.ndarray) -> int:
    """
    Baseline proporcional: sube si dz > 0 (goal arriba), baja si dz < 0.
    obs[0] = dz_norm (signed) — positivo si goal está arriba del EE actual.
    """
    dz_norm = obs[0]
    if dz_norm > 0.01:
        return ACTION_MOVE_UP
    elif dz_norm < -0.01:
        return ACTION_MOVE_DOWN
    else:
        return ACTION_ARM_STAY


# ---------------------------------------------------------------------------
# Función de evaluación
# ---------------------------------------------------------------------------

def evaluate(
    env: ArmEnv,
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
        avg_final_error_z   float (|dz| al terminar, todos los episodios)
        avg_efficiency      float (steps_min / steps_real, solo exitosos)
        per_episode         list de dicts
    """
    np.random.seed(seed)

    successes       = 0
    steps_list      = []
    final_errors    = []
    efficiency_list = []
    per_episode     = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        z_goal    = info["z_goal"]
        dz0       = info["initial_dz"]

        # Pasos mínimos teóricos: cuántos STEP_Z necesita para llegar
        min_steps = max(1, math.ceil(dz0 / STEP_Z))

        ep_steps   = 0
        ep_success = False
        final_dz   = dz0

        terminated = truncated = False
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_steps += 1
            final_dz  = step_info["dist_z"]

            if step_info.get("success", False):
                ep_success = True

        successes  += int(ep_success)
        final_errors.append(final_dz)

        ep_efficiency = (min_steps / ep_steps) if (ep_success and ep_steps > 0) else 0.0
        if ep_success:
            steps_list.append(ep_steps)
            efficiency_list.append(ep_efficiency)

        ep_result = {
            "episode":    ep + 1,
            "success":    ep_success,
            "steps":      ep_steps,
            "z_goal":     z_goal,
            "final_dz":   final_dz,
            "efficiency": ep_efficiency,
        }
        per_episode.append(ep_result)

        if verbose:
            tag = "✓" if ep_success else "✗"
            print(f"  [{tag}] Ep {ep+1:3d} | z_goal={z_goal:.3f} | "
                  f"steps={ep_steps:4d} | dz_final={final_dz:.4f} | eff={ep_efficiency:.3f}")

    success_rate    = successes / n_episodes
    avg_steps       = float(np.mean(steps_list))       if steps_list      else 0.0
    avg_final_error = float(np.mean(final_errors))
    avg_efficiency  = float(np.mean(efficiency_list))  if efficiency_list else 0.0

    return {
        "success_rate":      success_rate,
        "avg_steps":         avg_steps,
        "avg_final_error_z": avg_final_error,
        "avg_efficiency":    avg_efficiency,
        "per_episode":       per_episode,
    }


def print_summary(label: str, metrics: dict, n_episodes: int):
    print(f"\n{'='*52}")
    print(f"  {label}  ({n_episodes} episodios)")
    print(f"{'='*52}")
    print(f"  Tasa de éxito          : {metrics['success_rate']*100:.1f}%")
    print(f"  Steps promedio (éxito) : {metrics['avg_steps']:.1f}")
    print(f"  Error Z final (media)  : {metrics['avg_final_error_z']*100:.2f} cm")
    print(f"  Eficiencia promedio    : {metrics['avg_efficiency']:.3f}")
    print(f"{'='*52}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluación Agente 2 — posicionamiento Z (PyBullet)")
    p.add_argument("--model",    default=None,
                   help="Ruta al modelo .zip (SB3). Requerido si --baseline no se usa.")
    p.add_argument("--algo",     choices=["dqn", "ppo"], default="dqn",
                   help="Algoritmo del modelo cargado (default: dqn)")
    p.add_argument("--baseline", choices=["proportional"], default=None,
                   help="Evaluar política baseline")
    p.add_argument("--episodes", "--n_episodes", dest="n_episodes",
                   type=int, default=20,
                   help="Número de episodios de evaluación (default: 20)")
    p.add_argument("--gui",      action="store_true",
                   help="Abrir ventana gráfica de PyBullet")
    p.add_argument("--seed",     type=int, default=42,
                   help="Semilla para reproducibilidad (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.baseline is None and args.model is None:
        print("[evaluate_a2] Error: se debe indicar --model o --baseline.")
        return

    env = ArmEnv(gui=args.gui)

    if args.baseline == "proportional":
        label     = "Baseline Proporcional"
        policy_fn = _proportional_action
    else:
        from stable_baselines3 import DQN, PPO
        _cls = {"dqn": DQN, "ppo": PPO}[args.algo]
        model = _cls.load(args.model, env=env)
        label = f"{args.algo.upper()} ({os.path.basename(args.model)})"
        policy_fn = lambda obs: int(model.predict(obs, deterministic=True)[0])

    print(f"\n[evaluate_a2] {label} — {args.n_episodes} episodios  (seed={args.seed})")
    metrics = evaluate(env, policy_fn,
                       n_episodes=args.n_episodes, verbose=True, seed=args.seed)
    print_summary(label, metrics, args.n_episodes)

    env.close()


if __name__ == "__main__":
    main()
