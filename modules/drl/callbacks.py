"""
callbacks.py — Callbacks personalizados para el módulo DRL.
"""

from stable_baselines3.common.callbacks import EvalCallback


class StagnationEvalCallback(EvalCallback):
    """
    EvalCallback extendido con early stopping por estancamiento de reward.

    Detiene el entrenamiento si el reward promedio no mejora más de
    `min_delta` entre la primera y la mejor evaluación de la ventana de
    las últimas `patience` evaluaciones.

    Cuándo detiene:
        max(ventana) - ventana[0] < min_delta

    Esto captura dos casos:
      - Rewards planos (agente en loop, nunca converge).
      - Rewards decrecientes (agente desmejorando de forma sostenida).

    Parámetros adicionales sobre EvalCallback:
        patience  : número de evaluaciones consecutivas a observar antes de
                    poder activar el early stop (default: 5).
        min_delta : mejora mínima de reward requerida en la ventana para
                    considerar que hay aprendizaje (default: 1.0).
                    La escala depende del reward: con REWARD_SUCCESS=20 y
                    REWARD_COLLISION=-10, un delta de 1.0 es conservador.

    Ejemplo de uso en train.py:
        callback = StagnationEvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=best_dir,
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
            patience=5,
            min_delta=1.0,
        )
    """

    def __init__(self, *args, patience: int = 5, min_delta: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._patience: int = patience
        self._min_delta: float = min_delta
        self._eval_rewards: list = []

    def _on_step(self) -> bool:
        result = super()._on_step()

        # Registrar resultado cada vez que EvalCallback acaba de evaluar
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._eval_rewards.append(self.last_mean_reward)
            n = len(self._eval_rewards)

            if self.verbose >= 1:
                print(
                    f"[StagnationStop] Eval #{n:3d} "
                    f"@ {self.num_timesteps:,} steps — "
                    f"mean_reward={self.last_mean_reward:.3f}  "
                    f"(patience {n}/{self._patience})"
                )

            if n >= self._patience:
                window = self._eval_rewards[-self._patience:]
                improvement = max(window) - window[0]

                if improvement < self._min_delta:
                    if self.verbose >= 1:
                        print(
                            f"\n[StagnationStop] Deteniendo entrenamiento:\n"
                            f"  mejora en ventana = {improvement:.3f} "
                            f"< min_delta = {self._min_delta:.3f}\n"
                            f"  Últimas {self._patience} evals: "
                            f"{[f'{r:.2f}' for r in window]}\n"
                            f"  Steps: {self.num_timesteps:,}"
                        )
                    return False  # señal de parada a SB3

        return result
