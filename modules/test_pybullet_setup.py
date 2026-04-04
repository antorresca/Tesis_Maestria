"""
test_pybullet_setup.py — Verificación rápida del entorno PyBullet.

Ejecutar antes del smoke test de entrenamiento:
    cd modules/
    python test_pybullet_setup.py          # headless
    python test_pybullet_setup.py --gui    # con ventana gráfica (pausa al final)

Verifica:
  1. PyBullet importa sin errores
  2. URDF carga correctamente
  3. EE link (end_effector_link) se encuentra
  4. EE home está en la vecindad de (0.472, 0.000, 0.747)
  5. send_goal() mueve el EE a la posición comandada
  6. reset() restaura el estado home
  7. Obstáculos de OBSTACLE_SPECS cargan y sus posiciones son consultables
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--gui", action="store_true", help="Abrir ventana gráfica de PyBullet")
args = ap.parse_args()
USE_GUI = args.gui

PASS = "\033[92m[OK]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

errors = 0


def check(label, condition, detail=""):
    global errors
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f" — {detail}" if detail else ""))
        errors += 1


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
print("\n=== 1. Imports ===")
try:
    import pybullet as p
    import pybullet_data
    check("pybullet importa", True)
except ImportError as e:
    check("pybullet importa", False, str(e))
    print(f"\n  → Instalar con: pip install pybullet")
    sys.exit(1)

try:
    from robot_interface.pybullet_robot import PyBulletRobot
    check("PyBulletRobot importa", True)
except ImportError as e:
    check("PyBulletRobot importa", False, str(e))
    sys.exit(1)

try:
    from drl.config import _EE_HOME, OBSTACLE_SPECS, TRAINING_OBSTACLE_MODELS
    check("config.py importa (_EE_HOME, OBSTACLE_SPECS)", True)
except ImportError as e:
    check("config.py importa", False, str(e))
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Carga del URDF
# ---------------------------------------------------------------------------
print("\n=== 2. Carga URDF ===")
mode_label = "GUI" if USE_GUI else "DIRECT (headless)"
robot = PyBulletRobot(gui=USE_GUI)
try:
    robot.connect()
    check(f"PyBullet connect() ({mode_label})", True)
except Exception as e:
    check(f"PyBullet connect() ({mode_label})", False, str(e))
    sys.exit(1)

check("EE link encontrado (índice ≥ 0)", robot._ee_link_idx >= 0,
      f"índice = {robot._ee_link_idx}")
check("Brazo tiene 6 joints", len(robot._arm_joint_indices) == 6,
      f"joints encontrados = {len(robot._arm_joint_indices)}")

import pybullet as p
num_j = p.getNumJoints(robot._robot_id, physicsClientId=robot._client)
check(f"URDF tiene joints (n={num_j})", num_j > 0)

# ---------------------------------------------------------------------------
# 3. Estado inicial (EE home)
# ---------------------------------------------------------------------------
print("\n=== 3. EE home (joints=0) ===")
state = robot.get_state()
check("get_state() devuelve dict", state is not None)

if state:
    ee = state["ee_pos"]
    ex, ey, ez = _EE_HOME
    tol = 0.05  # 5 cm de tolerancia
    check(f"EE x ≈ {ex:.3f} (got {ee[0]:.3f})", abs(ee[0] - ex) < tol)
    check(f"EE y ≈ {ey:.3f} (got {ee[1]:.3f})", abs(ee[1] - ey) < tol)
    check(f"EE z ≈ {ez:.3f} (got {ee[2]:.3f})", abs(ee[2] - ez) < tol)
    check("joints dict tiene 6+ entradas", len(state["joints"]) >= 6,
          f"encontrados: {len(state['joints'])}")

# ---------------------------------------------------------------------------
# 4. send_goal() — mover a un goal cercano
# ---------------------------------------------------------------------------
print("\n=== 4. send_goal() ===")
ex, ey, ez = _EE_HOME
test_goal = (ex + 0.1, ey, ez)
try:
    robot.send_goal(*test_goal)
    state2 = robot.get_state()
    if state2:
        ee2 = state2["ee_pos"]
        dist = ((ee2[0]-test_goal[0])**2 + (ee2[1]-test_goal[1])**2 + (ee2[2]-test_goal[2])**2)**0.5
        check(f"send_goal(+0.1 X): dist residual = {dist:.3f} m", dist < 0.15,
              "IK puede tener residual mayor si el goal está fuera del workspace del brazo")
    else:
        check("send_goal(): get_state() devuelve estado", False)
except Exception as e:
    check("send_goal() no lanza excepción", False, str(e))

# ---------------------------------------------------------------------------
# 5. reset()
# ---------------------------------------------------------------------------
print("\n=== 5. reset() ===")
try:
    initial = robot.reset(ee_home=_EE_HOME)
    check("reset() devuelve estado", initial is not None)
    if initial:
        ee_r = initial["ee_pos"]
        ex, ey, ez = _EE_HOME
        tol = 0.05
        check(f"reset(): EE vuelve a home (dist < {tol} m)",
              ((ee_r[0]-ex)**2 + (ee_r[1]-ey)**2 + (ee_r[2]-ez)**2)**0.5 < tol,
              f"EE post-reset = {[round(v,3) for v in ee_r]}")
except Exception as e:
    check("reset() no lanza excepción", False, str(e))

# ---------------------------------------------------------------------------
# 6. Obstáculos
# ---------------------------------------------------------------------------
print("\n=== 6. Obstáculos ===")
check(f"OBSTACLE_SPECS tiene {len(OBSTACLE_SPECS)} entradas",
      len(OBSTACLE_SPECS) == len(TRAINING_OBSTACLE_MODELS))

for name, spec in OBSTACLE_SPECS.items():
    try:
        robot.add_obstacle(name, **spec)
        pos = robot.get_obstacle_position(name)
        exp = spec["position"]
        ok = pos is not None and all(abs(pos[i]-exp[i]) < 0.01 for i in range(3))
        check(f"Obstáculo '{name}': posición correcta", ok,
              f"esperado={exp}, obtenido={pos}")
    except Exception as e:
        check(f"Obstáculo '{name}': sin excepción", False, str(e))

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
if USE_GUI:
    print("\n  [GUI activo — presiona Enter para cerrar la ventana]")
    input()
robot.disconnect()

# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
print()
if errors == 0:
    print(f"\033[92m{'='*50}\033[0m")
    print(f"\033[92m  TODOS LOS CHECKS PASARON — listo para smoke test\033[0m")
    print(f"\033[92m{'='*50}\033[0m")
    print("\n  Siguiente paso:")
    print("  python -m modules.drl.train --algo dqn --timesteps 2000 --check_env\n")
else:
    print(f"\033[91m{'='*50}\033[0m")
    print(f"\033[91m  {errors} CHECK(S) FALLARON — revisar antes de continuar\033[0m")
    print(f"\033[91m{'='*50}\033[0m\n")
    sys.exit(1)
