"""
train.py — Entrenamiento DQN / PPO sobre MobileManipulatorEnv.

Uso:
    # DQN (principal)
    python -m modules.drl.train --algo dqn

    # PPO (comparación académica)
    python -m modules.drl.train --algo ppo --timesteps 500000

    # Con obstáculos y rosbridge remoto
    python -m modules.drl.train --algo dqn --host 192.168.1.10 \\
        --obstacle_models unit_box_1 unit_box_2

Salida:
    models/drl/<algo>_model.zip         modelo entrenado (SB3)
    models/drl/logs/<algo>/             tensorboard logs
"""

import argparse
import os
import sys

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, CallbackList, ProgressBarCallback
)

try:
    from .mobile_manipulator_env import MobileManipulatorEnv
    from .config import DQN_HPARAMS, PPO_HPARAMS, TOTAL_TIMESTEPS, TRAINING_OBSTACLE_MODELS
    from .callbacks import StagnationEvalCallback
except ImportError:
    from mobile_manipulator_env import MobileManipulatorEnv
    from config import DQN_HPARAMS, PPO_HPARAMS, TOTAL_TIMESTEPS, TRAINING_OBSTACLE_MODELS
    from callbacks import StagnationEvalCallback


_ALGO_MAP = {
    "dqn": DQN,
    "ppo": PPO,
}

_HPARAMS_MAP = {
    "dqn": DQN_HPARAMS,
    "ppo": PPO_HPARAMS,
}


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento DRL — robot manipulador móvil")
    p.add_argument("--algo",       choices=["dqn", "ppo"], default="dqn",
                   help="Algoritmo a entrenar (default: dqn)")
    p.add_argument("--host",       default="localhost",
                   help="IP del rosbridge (default: localhost)")
    p.add_argument("--port",       type=int, default=9090,
                   help="Puerto rosbridge (default: 9090)")
    p.add_argument("--timesteps",  type=int, default=TOTAL_TIMESTEPS,
                   help=f"Total de timesteps (default: {TOTAL_TIMESTEPS})")
    p.add_argument("--save_dir",   default="models/drl",
                   help="Directorio de salida para modelos y logs")
    p.add_argument("--obstacle_models", nargs="*", default=None,
                   help="Model names de Gazebo (default: TRAINING_OBSTACLE_MODELS de config.py)")
    p.add_argument("--check_env",  action="store_true",
                   help="Ejecutar check_env de SB3 antes de entrenar y salir")
    p.add_argument("--eval_freq",  type=int, default=10_000,
                   help="Cada cuántos steps evaluar el modelo (default: 10000)")
    p.add_argument("--n_eval_eps", type=int, default=5,
                   help="Episodios por evaluación (default: 5)")
    p.add_argument("--checkpoint_freq", type=int, default=50_000,
                   help="Cada cuántos steps guardar checkpoint (default: 50000)")
    # Early stopping por estancamiento
    p.add_argument("--patience",  type=int,   default=5,
                   help="Evaluaciones sin mejora para activar early stop (default: 5)")
    p.add_argument("--min_delta", type=float, default=1.0,
                   help="Mejora mínima de reward en la ventana patience (default: 1.0)")
    return p.parse_args()


def main():
    args = parse_args()
    algo_name = args.algo

    # Si no se pasaron obstáculos por CLI, usar el default de config.py
    obstacle_models = args.obstacle_models if args.obstacle_models is not None \
                      else TRAINING_OBSTACLE_MODELS

    print(f"[train] Algoritmo: {algo_name.upper()}")
    print(f"[train] Rosbridge: {args.host}:{args.port}")
    print(f"[train] Timesteps: {args.timesteps:,}")
    print(f"[train] Obstáculos: {obstacle_models or 'ninguno'}")

    # ---- Rutas de salida ------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir   = os.path.join(args.save_dir, "logs", algo_name)
    best_dir  = os.path.join(args.save_dir, "best", algo_name)
    ckpt_dir  = os.path.join(args.save_dir, "checkpoints", algo_name)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Entorno --------------------------------------------------------
    env = MobileManipulatorEnv(
        host=args.host,
        port=args.port,
        obstacle_models=obstacle_models,
    )

    if args.check_env:
        print("[train] Ejecutando check_env …")
        check_env(env, warn=True)
        print("[train] check_env OK — saliendo.")
        env.close()
        return

    # ---- Modelo ---------------------------------------------------------
    AlgoClass = _ALGO_MAP[algo_name]
    hparams   = _HPARAMS_MAP[algo_name]

    try:
        import tensorboard  # noqa: F401
        tb_log = log_dir
    except ImportError:
        tb_log = None
        print("[train] tensorboard no instalado — logs desactivados. "
              "Instalar con: pip install tensorboard")

    model = AlgoClass(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tb_log,
        **hparams,
    )

    # ---- Callbacks ------------------------------------------------------
    eval_env = MobileManipulatorEnv(
        host=args.host,
        port=args.port,
        obstacle_models=obstacle_models,
    )

    callbacks = CallbackList([
        ProgressBarCallback(),
        StagnationEvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=best_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_eps,
            deterministic=True,
            verbose=1,
            patience=args.patience,
            min_delta=args.min_delta,
        ),
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=ckpt_dir,
            name_prefix=algo_name,
            verbose=1,
        ),
    ])

    # ---- Entrenamiento --------------------------------------------------
    print(f"[train] Iniciando entrenamiento …")
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[train] Entrenamiento interrumpido por el usuario.")

    # ---- Guardar modelo final -------------------------------------------
    final_path = os.path.join(args.save_dir, f"{algo_name}_model")
    model.save(final_path)
    print(f"[train] Modelo final guardado en {final_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
