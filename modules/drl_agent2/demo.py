"""
demo.py — Visualización interactiva del Agente 2 entrenado (posicionamiento Z).

Corre episodios paso a paso con delay configurable para ver el movimiento
del brazo subiendo/bajando hasta el z_goal.

Uso:
    # Goal Z específico
    python -m modules.drl_agent2.demo --model models/drl_agent2/best/dqn/best_model --z 0.5

    # Goal Z aleatorio (varios episodios)
    python -m modules.drl_agent2.demo --model models/drl_agent2/best/dqn/best_model --episodes 5

    # Con delay más lento
    python -m modules.drl_agent2.demo --model models/drl_agent2/best/dqn/best_model --delay 0.3

    # Loop continuo
    python -m modules.drl_agent2.demo --model models/drl_agent2/best/dqn/best_model --loop

    # Baseline proporcional (sin modelo)
    python -m modules.drl_agent2.demo --baseline proportional --episodes 3
"""

import argparse
import time

import numpy as np

try:
    from .arm_env import ArmEnv
    from .config import ARM_GOAL_SAMPLE_Z, GOAL_Z_THR, ARM_Z_SPAN
    from ..drl.config import ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_ARM_STAY, WORKSPACE_Z
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from drl_agent2.arm_env import ArmEnv
    from drl_agent2.config import ARM_GOAL_SAMPLE_Z, GOAL_Z_THR, ARM_Z_SPAN
    from drl.config import ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_ARM_STAY, WORKSPACE_Z

ACTION_NAMES = {
    ACTION_MOVE_UP:   "MOVE_UP  ",
    ACTION_MOVE_DOWN: "MOVE_DOWN",
    ACTION_ARM_STAY:  "STAY     ",
}


def _proportional_action(obs):
    dz_norm = obs[0]
    if dz_norm > 0.01:
        return ACTION_MOVE_UP
    elif dz_norm < -0.01:
        return ACTION_MOVE_DOWN
    return ACTION_ARM_STAY


def run_episode(env, policy_fn, z_goal: float, delay: float, verbose: bool = True) -> dict:
    """Corre un episodio con delay entre steps para visualización."""
    obs, info = env.reset(options={"z_goal": z_goal})

    if verbose:
        dz0 = info["initial_dz"]
        print(f"\n  z_goal={z_goal:.3f} m  |  dz_inicial={dz0:.4f} m")
        print(f"  {'step':>5}  {'action':<12}  {'ee_z':>7}  {'dz':>8}  {'reward':>7}  status")
        print(f"  {'-'*60}")

    ep_steps   = 0
    ep_success = False

    terminated = truncated = False
    while not (terminated or truncated):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, step_info = env.step(action)

        ep_steps += 1
        if step_info.get("success", False):
            ep_success = True

        if verbose:
            ee_z_approx = obs[1] * ARM_Z_SPAN + WORKSPACE_Z[0]

            status = ""
            if step_info.get("success"):
                status = "SUCCESS ✓"
            elif step_info.get("collision"):
                status = "COLLISION ✗"
            elif truncated:
                status = "TIMEOUT ✗"

            dz = step_info["dz"]
            print(f"  {ep_steps:>5}  {ACTION_NAMES[action]}   "
                  f"{ee_z_approx:>7.4f}  {dz:>+8.4f}  {reward:>7.3f}  {status}")

        time.sleep(delay)

    return {"success": ep_success, "steps": ep_steps}


def parse_args():
    p = argparse.ArgumentParser(description="Demo Agente 2 — posicionamiento Z (PyBullet GUI)")
    p.add_argument("--model",    default=None,
                   help="Ruta al modelo SB3 (.zip sin extensión)")
    p.add_argument("--algo",     choices=["dqn", "ppo"], default="dqn",
                   help="Algoritmo del modelo cargado (default: dqn)")
    p.add_argument("--baseline", choices=["proportional"], default=None,
                   help="Usar política baseline en lugar del modelo")
    p.add_argument("--z",        type=float, default=None,
                   help="Altura Z goal específica (m). Si no se da, se muestrea aleatoriamente.")
    p.add_argument("--episodes", type=int, default=3,
                   help="Número de episodios a visualizar (default: 3)")
    p.add_argument("--delay",    type=float, default=0.1,
                   help="Segundos entre steps (default: 0.1)")
    p.add_argument("--loop",     action="store_true",
                   help="Repetir episodios en bucle hasta Ctrl+C")
    return p.parse_args()


def main():
    args = parse_args()

    if args.baseline is None and args.model is None:
        print("[demo_a2] Error: se debe indicar --model o --baseline.")
        return

    env = ArmEnv(gui=True)

    # ---- Política -------------------------------------------------------
    if args.baseline == "proportional":
        label     = "Baseline Proporcional"
        policy_fn = _proportional_action
    else:
        from stable_baselines3 import DQN, PPO
        _cls = {"dqn": DQN, "ppo": PPO}[args.algo]
        model = _cls.load(args.model, env=env)
        label = f"{args.algo.upper()} — {args.model}"
        policy_fn = lambda obs: int(model.predict(obs, deterministic=True)[0])

    print(f"\n[demo_a2] {label}")
    print(f"[demo_a2] delay={args.delay}s | loop={'sí' if args.loop else 'no'}")

    # ---- Episodios -------------------------------------------------------
    z_goals = []
    if args.z is not None:
        z_goals = [args.z] * args.episodes
    else:
        z_goals = list(np.random.uniform(*ARM_GOAL_SAMPLE_Z, size=args.episodes))

    ep_idx = 0
    try:
        while True:
            z = z_goals[ep_idx % len(z_goals)]
            result = run_episode(env, policy_fn, z_goal=z, delay=args.delay)
            tag = "✓" if result["success"] else "✗"
            print(f"  → Episodio {ep_idx+1}: [{tag}] {result['steps']} steps")

            ep_idx += 1
            if not args.loop and ep_idx >= args.episodes:
                break
    except KeyboardInterrupt:
        print("\n[demo_a2] Interrumpido por el usuario.")

    env.close()


if __name__ == "__main__":
    main()
