"""
train.py — Entrenamiento DQN / PPO sobre ArmEnv (Agente 2, posicionamiento Z).

Uso:
    # DQN
    python -m modules.drl_agent2.train --algo dqn

    # PPO (comparación)
    python -m modules.drl_agent2.train --algo ppo --timesteps 300000

    # Con GUI de PyBullet
    python -m modules.drl_agent2.train --algo dqn --gui

    # Smoke test
    python -m modules.drl_agent2.train --algo dqn --timesteps 1000 --check_env

Salida:
    models/drl_agent2/<algo>_model.zip         modelo entrenado (SB3)
    models/drl_agent2/logs/<algo>/             tensorboard logs
    models/drl_agent2/best/<algo>/best_model   mejor modelo (EvalCallback)
"""

import argparse
import os

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, CallbackList, ProgressBarCallback
)

try:
    from .arm_env import ArmEnv
    from .config import ARM_DQN_HPARAMS, ARM_PPO_HPARAMS, ARM_TOTAL_TIMESTEPS
    from ..drl.callbacks import StagnationEvalCallback
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from drl_agent2.arm_env import ArmEnv
    from drl_agent2.config import ARM_DQN_HPARAMS, ARM_PPO_HPARAMS, ARM_TOTAL_TIMESTEPS
    from drl.callbacks import StagnationEvalCallback


_ALGO_MAP = {
    "dqn": DQN,
    "ppo": PPO,
}

_HPARAMS_MAP = {
    "dqn": ARM_DQN_HPARAMS,
    "ppo": ARM_PPO_HPARAMS,
}


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento Agente 2 — posicionamiento Z (PyBullet)")
    p.add_argument("--algo",       choices=["dqn", "ppo"], default="dqn",
                   help="Algoritmo a entrenar (default: dqn)")
    p.add_argument("--gui",        action="store_true",
                   help="Abrir ventana gráfica de PyBullet durante el entrenamiento")
    p.add_argument("--timesteps",  type=int, default=ARM_TOTAL_TIMESTEPS,
                   help=f"Total de timesteps (default: {ARM_TOTAL_TIMESTEPS})")
    p.add_argument("--save_dir",   default="models/drl_agent2",
                   help="Directorio de salida para modelos y logs")
    p.add_argument("--check_env",  action="store_true",
                   help="Ejecutar check_env de SB3 antes de entrenar y salir")
    p.add_argument("--eval_freq",  type=int, default=5_000,
                   help="Cada cuántos steps evaluar el modelo (default: 5000)")
    p.add_argument("--n_eval_eps", type=int, default=10,
                   help="Episodios por evaluación (default: 10)")
    p.add_argument("--checkpoint_freq", type=int, default=25_000,
                   help="Cada cuántos steps guardar checkpoint (default: 25000)")
    p.add_argument("--patience",   type=int,   default=10,
                   help="Evaluaciones sin mejora para early stop (default: 10)")
    p.add_argument("--min_delta",  type=float, default=0.5,
                   help="Mejora mínima de reward en la ventana patience (default: 0.5)")
    return p.parse_args()


def main():
    args = parse_args()
    algo_name = args.algo

    print(f"[train_a2] Algoritmo: {algo_name.upper()}")
    print(f"[train_a2] Simulador: PyBullet {'GUI' if args.gui else 'DIRECT (headless)'}")
    print(f"[train_a2] Timesteps: {args.timesteps:,}")

    # ---- Rutas de salida ------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir  = os.path.join(args.save_dir, "logs", algo_name)
    best_dir = os.path.join(args.save_dir, "best", algo_name)
    ckpt_dir = os.path.join(args.save_dir, "checkpoints", algo_name)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Entorno --------------------------------------------------------
    env = ArmEnv(gui=args.gui)

    if args.check_env:
        print("[train_a2] Ejecutando check_env …")
        check_env(env, warn=True)
        print("[train_a2] check_env OK — saliendo.")
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
        print("[train_a2] tensorboard no instalado — logs desactivados.")

    model = AlgoClass(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tb_log,
        **hparams,
    )

    # ---- Callbacks ------------------------------------------------------
    eval_env = ArmEnv(gui=False)

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
    print("[train_a2] Iniciando entrenamiento …")
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[train_a2] Entrenamiento interrumpido por el usuario.")

    # ---- Guardar modelo final -------------------------------------------
    final_path = os.path.join(args.save_dir, f"{algo_name}_model")
    model.save(final_path)
    print(f"[train_a2] Modelo final guardado en {final_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
