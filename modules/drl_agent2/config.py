"""
config.py — Hiperparámetros del Agente 2 (posicionamiento Z del brazo).

Importa las constantes compartidas de drl/config.py y agrega las
específicas del brazo. No duplicar valores ya definidos allí.
"""

# ---------------------------------------------------------------------------
# Workspace Z (rango válido del efector)
# ---------------------------------------------------------------------------
# Rango de muestreo de goals Z: dentro del WORKSPACE_Z con margen para
# evitar poses extremas donde el IK tiende a fallar.
ARM_GOAL_SAMPLE_Z = (0.40, 1.10)   # m — goals Z durante entrenamiento

# Umbral de éxito en Z — más ajustado que el Agente 1 (brazo = precisión)
GOAL_Z_THR = 0.05   # m

# Span total del workspace Z — usado para normalización de observaciones
# Equivale a WORKSPACE_Z[1] - WORKSPACE_Z[0] = 1.20 - 0.35 = 0.85 m
ARM_Z_SPAN = 0.85   # m

# Máximo de steps por episodio — tarea 1D, convergencia rápida
ARM_MAX_STEPS = 150

# Timesteps totales de entrenamiento
ARM_TOTAL_TIMESTEPS = 300_000

# ---------------------------------------------------------------------------
# Hiperparámetros SB3 — DQN (Agente 2)
# ---------------------------------------------------------------------------
ARM_DQN_HPARAMS = {
    "learning_rate":           1e-4,
    "buffer_size":             50_000,
    "learning_starts":         500,
    "batch_size":              64,
    "tau":                     1.0,
    "gamma":                   0.99,
    "train_freq":              4,
    "gradient_steps":          1,
    "target_update_interval":  500,
    "exploration_fraction":    0.20,   # explora hasta step 60k de 300k
    "exploration_final_eps":   0.05,
    "policy_kwargs":           {"net_arch": [64, 64]},   # red más pequeña — problema 1D
}

# ---------------------------------------------------------------------------
# Hiperparámetros SB3 — PPO (Agente 2)
# ---------------------------------------------------------------------------
ARM_PPO_HPARAMS = {
    "learning_rate":  3e-4,
    "n_steps":        2048,
    "batch_size":     64,
    "n_epochs":       10,
    "gamma":          0.99,
    "gae_lambda":     0.95,
    "clip_range":     0.2,
    "ent_coef":       0.01,
    "policy_kwargs":  {"net_arch": [64, 64]},
}
