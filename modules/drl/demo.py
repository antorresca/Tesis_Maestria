"""
demo.py — Visualización interactiva del agente DRL entrenado (PyBullet GUI).

Corre episodios paso a paso con delay configurable para ver el movimiento
del robot. Permite especificar goals manualmente o usar los goals de entrenamiento.

Uso:
    # Goal por defecto (forward 0.6m)
    python -m modules.drl.demo --model models/drl/best/dqn/best_model

    # Goal específico
    python -m modules.drl.demo --model models/drl/best/dqn/best_model --x 1.5 --y 0.5

    # Todos los goals de entrenamiento, 0.3s entre steps
    python -m modules.drl.demo --model models/drl/best/dqn/best_model --all_goals --delay 0.3

    # Loop continuo (repite episodios)
    python -m modules.drl.demo --model models/drl/best/dqn/best_model --all_goals --loop
"""

import argparse
import math
import time

import numpy as np

try:
    from .mobile_manipulator_env import MobileManipulatorEnv
    from .continuous_env import ContinuousManipulatorEnv
    from .config import DEFAULT_TRAINING_GOALS, TRAINING_OBSTACLE_MODELS, _EE_HOME, GOAL_THR
except ImportError:
    from mobile_manipulator_env import MobileManipulatorEnv
    from continuous_env import ContinuousManipulatorEnv
    from config import DEFAULT_TRAINING_GOALS, TRAINING_OBSTACLE_MODELS, _EE_HOME, GOAL_THR


def run_episode(env, policy_fn, pose_goal: dict, delay: float, verbose: bool = True) -> dict:
    """Corre un episodio con delay entre steps para visualización."""
    obs, info = env.reset(options={"pose_goal": pose_goal})

    goal_x = pose_goal["x"]
    goal_y = pose_goal["y"]
    goal_z = pose_goal["z"]

    if verbose:
        dist0 = info["initial_dist"]
        print(f"\n  Goal: ({goal_x:.2f}, {goal_y:.2f}, {goal_z:.2f})  "
              f"dist_inicial={dist0:.3f}m")
        print(f"  {'step':>5}  {'action':<12}  {'dist':>6}  {'reward':>7}  status")
        print(f"  {'-'*55}")

    ACTION_NAMES = {0: "FORWARD", 1: "TURN_LEFT", 2: "TURN_RIGHT",
                    3: "BACKWARD", 4: "STAY"}

    ep_steps = 0
    ep_collisions = 0
    ep_success = False

    terminated = truncated = False
    while not (terminated or truncated):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, step_info = env.step(action)

        ep_steps += 1
        if step_info.get("collision", False):
            ep_collisions += 1

        if step_info.get("success", False):
            ep_success = True

        if verbose:
            dist    = step_info["dist_goal"]
            status  = ""
            if step_info.get("success"):
                status = "SUCCESS ✓"
            elif step_info.get("collision"):
                status = "COLLISION ✗"
            elif truncated:
                status = "TIMEOUT ✗"
            print(f"  {ep_steps:>5}  {ACTION_NAMES[action]:<12}  {dist:>6.3f}  {reward:>7.3f}  {status}")

        time.sleep(delay)

    return {
        "success":    ep_success,
        "steps":      ep_steps,
        "collisions": ep_collisions,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Demo DRL — visualización interactiva (PyBullet GUI)")
    p.add_argument("--model",      required=True,
                   help="Ruta al modelo .zip (SB3), sin extensión")
    p.add_argument("--algo",       choices=["dqn", "ppo", "sac"], default="dqn")
    p.add_argument("--continuous", action="store_true",
                   help="Usar ContinuousManipulatorEnv (auto si --algo sac)")
    p.add_argument("--delay",      type=float, default=0.15,
                   help="Segundos de pausa entre steps (default: 0.15)")

    # Goal manual
    p.add_argument("--x", type=float, default=None, help="Goal EE x (m)")
    p.add_argument("--y", type=float, default=None, help="Goal EE y (m)")
    p.add_argument("--z", type=float, default=None,
                   help=f"Goal EE z (m), default: {_EE_HOME[2]}")

    # Modo multi-goal
    p.add_argument("--all_goals", action="store_true",
                   help="Recorrer todos los DEFAULT_TRAINING_GOALS en orden")
    p.add_argument("--loop", action="store_true",
                   help="Repetir episodios indefinidamente (Ctrl+C para salir)")

    p.add_argument("--obstacle_models", nargs="*", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    continuous = args.continuous or (args.algo == "sac")

    obstacle_models = args.obstacle_models if args.obstacle_models is not None \
                      else TRAINING_OBSTACLE_MODELS

    modo = "continuo" if continuous else "discreto"
    print(f"[demo] Cargando modelo: {args.model}  ({args.algo.upper()} {modo})")
    print(f"[demo] Delay entre steps: {args.delay}s")
    print(f"[demo] Obstáculos: {obstacle_models or 'ninguno'}")

    EnvClass = ContinuousManipulatorEnv if continuous else MobileManipulatorEnv
    env = EnvClass(gui=True, obstacle_models=obstacle_models)

    from stable_baselines3 import DQN, PPO, SAC
    _cls = {"dqn": DQN, "ppo": PPO, "sac": SAC}[args.algo]
    model = _cls.load(args.model, env=env)
    policy_fn = lambda obs: model.predict(obs, deterministic=True)[0]

    # Construir lista de goals
    if args.all_goals:
        goals = DEFAULT_TRAINING_GOALS
    elif args.x is not None or args.y is not None:
        goals = [{
            "x":  args.x if args.x is not None else _EE_HOME[0],
            "y":  args.y if args.y is not None else _EE_HOME[1],
            "z":  args.z if args.z is not None else _EE_HOME[2],
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
            "intent": "navigate",
        }]
    else:
        # Goal por defecto: el primero de entrenamiento
        goals = [DEFAULT_TRAINING_GOALS[0]]

    print(f"[demo] {len(goals)} goal(s) a recorrer. Ctrl+C para salir.\n")

    iteration = 0
    try:
        while True:
            iteration += 1
            if args.loop and len(goals) > 1:
                print(f"[demo] === Vuelta {iteration} ===")

            for i, goal in enumerate(goals):
                label = f"Goal {i+1}/{len(goals)}"
                print(f"[demo] {label}")
                result = run_episode(env, policy_fn, goal, delay=args.delay)
                tag = "✓ éxito" if result["success"] else "✗ fallo"
                print(f"  → {tag} en {result['steps']} steps, {result['collisions']} colisiones")

                if i < len(goals) - 1:
                    print("  [pausa 1s antes del siguiente goal]")
                    time.sleep(1.0)

            if not args.loop:
                break

    except KeyboardInterrupt:
        print("\n[demo] Interrumpido.")

    env.close()


if __name__ == "__main__":
    main()
