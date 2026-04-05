"""
train.py — Entrenamiento DQN / PPO (discreto) y SAC / PPO (continuo) sobre PyBullet.

Uso:
    # DQN discreto (original)
    python -m modules.drl.train --algo dqn

    # PPO discreto
    python -m modules.drl.train --algo ppo --timesteps 1000000

    # SAC continuo
    python -m modules.drl.train --algo sac --continuous

    # PPO continuo
    python -m modules.drl.train --algo ppo --continuous

    # Con GUI
    python -m modules.drl.train --algo sac --continuous --gui

    # Smoke test
    python -m modules.drl.train --algo sac --continuous --timesteps 2000 --check_env

Salida:
    models/drl/<algo>_model.zip         modelo entrenado (SB3)
    models/drl/logs/<algo>/             tensorboard logs
"""

import argparse
import os

from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, CallbackList, ProgressBarCallback
)

try:
    from .mobile_manipulator_env import MobileManipulatorEnv
    from .continuous_env import ContinuousManipulatorEnv
    from .config import (
        DQN_HPARAMS, PPO_HPARAMS, SAC_HPARAMS, PPO_CONT_HPARAMS,
        TOTAL_TIMESTEPS, TRAINING_OBSTACLE_MODELS,
    )
    from .callbacks import StagnationEvalCallback
except ImportError:
    from mobile_manipulator_env import MobileManipulatorEnv
    from continuous_env import ContinuousManipulatorEnv
    from config import (
        DQN_HPARAMS, PPO_HPARAMS, SAC_HPARAMS, PPO_CONT_HPARAMS,
        TOTAL_TIMESTEPS, TRAINING_OBSTACLE_MODELS,
    )
    from callbacks import StagnationEvalCallback


# Algoritmos disponibles por modo
_ALGO_MAP = {
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
}

# Hiperparámetros: (discreto, continuo)
_HPARAMS_MAP = {
    "dqn": (DQN_HPARAMS,      None),           # solo discreto
    "ppo": (PPO_HPARAMS,      PPO_CONT_HPARAMS),
    "sac": (None,             SAC_HPARAMS),     # solo continuo
}


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento DRL — robot manipulador móvil (PyBullet)")
    p.add_argument("--algo",       choices=["dqn", "ppo", "sac"], default="dqn",
                   help="Algoritmo a entrenar (default: dqn)")
    p.add_argument("--continuous", action="store_true",
                   help="Usar espacio de acción continuo Box(2) (requerido para sac)")
    p.add_argument("--gui",        action="store_true",
                   help="Abrir ventana gráfica de PyBullet durante el entrenamiento")
    p.add_argument("--timesteps",  type=int, default=TOTAL_TIMESTEPS,
                   help=f"Total de timesteps (default: {TOTAL_TIMESTEPS})")
    p.add_argument("--save_dir",   default="models/drl",
                   help="Directorio de salida para modelos y logs")
    p.add_argument("--obstacle_models", nargs="*", default=None,
                   help="Model names de obstáculos (default: TRAINING_OBSTACLE_MODELS de config.py)")
    p.add_argument("--check_env",  action="store_true",
                   help="Ejecutar check_env de SB3 antes de entrenar y salir")
    p.add_argument("--eval_freq",  type=int, default=10_000,
                   help="Cada cuántos steps evaluar el modelo (default: 10000)")
    p.add_argument("--n_eval_eps", type=int, default=5,
                   help="Episodios por evaluación (default: 5)")
    p.add_argument("--checkpoint_freq", type=int, default=50_000,
                   help="Cada cuántos steps guardar checkpoint (default: 50000)")
    p.add_argument("--patience",  type=int,   default=5,
                   help="Evaluaciones sin mejora para activar early stop (default: 5)")
    p.add_argument("--min_delta", type=float, default=1.0,
                   help="Mejora mínima de reward en la ventana patience (default: 1.0)")
    return p.parse_args()


def main():
    args = parse_args()
    algo_name  = args.algo
    continuous = args.continuous or (algo_name == "sac")   # SAC implica continuo

    # Validar combinación algo/modo
    hparams_disc, hparams_cont = _HPARAMS_MAP[algo_name]
    if continuous and hparams_cont is None:
        print(f"[train] Error: {algo_name.upper()} solo soporta modo discreto.")
        return
    if not continuous and hparams_disc is None:
        print(f"[train] Error: {algo_name.upper()} requiere --continuous.")
        return

    hparams = hparams_cont if continuous else hparams_disc

    obstacle_models = args.obstacle_models if args.obstacle_models is not None \
                      else TRAINING_OBSTACLE_MODELS

    modo = "continuo" if continuous else "discreto"
    print(f"[train] Algoritmo: {algo_name.upper()} ({modo})")
    print(f"[train] Simulador: PyBullet {'GUI' if args.gui else 'DIRECT (headless)'}")
    print(f"[train] Timesteps: {args.timesteps:,}")
    print(f"[train] Obstáculos: {obstacle_models or 'ninguno'}")

    # ---- Rutas de salida — sufijo para no mezclar con discreto ----------
    save_tag = f"{algo_name}_cont" if continuous else algo_name
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir  = os.path.join(args.save_dir, "logs",        save_tag)
    best_dir = os.path.join(args.save_dir, "best",        save_tag)
    ckpt_dir = os.path.join(args.save_dir, "checkpoints", save_tag)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Entorno --------------------------------------------------------
    EnvClass = ContinuousManipulatorEnv if continuous else MobileManipulatorEnv
    env = EnvClass(gui=args.gui, obstacle_models=obstacle_models)

    if args.check_env:
        print("[train] Ejecutando check_env …")
        check_env(env, warn=True)
        print("[train] check_env OK — saliendo.")
        env.close()
        return

    # ---- Modelo ---------------------------------------------------------
    AlgoClass = _ALGO_MAP[algo_name]

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
    eval_env = EnvClass(gui=False, obstacle_models=obstacle_models)

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
    # PyBullet es síncrono — no se necesita sleep de inicialización.
    print("[train] Iniciando entrenamiento …")
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[train] Entrenamiento interrumpido por el usuario.")

    # ---- Guardar modelo final -------------------------------------------
    final_path = os.path.join(args.save_dir, f"{save_tag}_model")
    model.save(final_path)
    print(f"[train] Modelo final guardado en {final_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
