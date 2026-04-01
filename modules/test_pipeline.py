"""
test_pipeline.py
----------------
Prueba de integración: PLN → Goal Builder

Modos:
  --pln-only   Solo prueba el PLN (no requiere Gazebo).
               Útil para verificar clasificación y extracción de entidades.

  (default)    Pipeline completo: PLN → GoalBuilder → pose_goal
               Requiere Gazebo corriendo con training_world y rosbridge en :9090.

Uso:
    # Modo rápido (sin Gazebo):
    python test_pipeline.py --pln-only

    # Pipeline completo (con Gazebo):
    python test_pipeline.py

    # Con frases personalizadas:
    python test_pipeline.py --pln-only --texts "ve a la mesa" "recoge el cubo rojo"
    python test_pipeline.py --texts "ve a la mesa" "recoge el cubo rojo"

    # Gazebo en otra máquina:
    python test_pipeline.py --host 192.168.1.X
"""

import sys
import json
import argparse
from pathlib import Path

# --- Rutas de importación ---
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "pln" / "src"))
sys.path.insert(0, str(ROOT / "goal_builder"))

# Frases de prueba que cubren los 3 intents y ambos idiomas
DEFAULT_TEXTS = [
    # navigate
    "ve a la mesa",
    "go to the green zone",
    "navega hacia la zona verde",
    "go home",
    "regresa a casa",
    # pick
    "recoge el cubo rojo",
    "pick up the red cube",
    "toma el cubo de la mesa",
    # place
    "deja el cubo en la bandeja",
    "place the cube on the tray",
    "lleva el objeto a la bandeja",
]

SEP = "─" * 60


def run_pln_only(texts: list[str], model_type: str):
    from pln_module import PLNModule

    pln = PLNModule(model_type=model_type)
    print(f"\n{SEP}")
    print("MODO: solo PLN (sin Gazebo)")
    print(SEP)

    for text in texts:
        result = pln.predict(text)
        print(f"\n  Texto : {text}")
        print(f"  Intent: {result['intent']}  (conf={result['confidence']:.3f})")
        print(f"  target: {result['target']}   destination: {result['destination']}")

    print(f"\n{SEP}\n")


def run_full_pipeline(texts: list[str], model_type: str, host: str, port: int):
    from pln_module import PLNModule
    from goal_builder import GoalBuilder

    pln = PLNModule(model_type=model_type)
    gb  = GoalBuilder(host=host, port=port)
    gb.connect()

    print(f"\n{SEP}")
    print(f"MODO: pipeline completo (rosbridge {host}:{port})")
    print(SEP)

    ok = 0
    fail = 0

    for text in texts:
        pln_result = pln.predict(text)
        print(f"\n  Texto : {text}")
        print(f"  PLN   : intent={pln_result['intent']}  "
              f"target={pln_result['target']}  "
              f"destination={pln_result['destination']}  "
              f"conf={pln_result['confidence']:.3f}")

        try:
            pose_goal = gb.build(pln_result)
            print(f"  Goal  : x={pose_goal['x']:.3f}  y={pose_goal['y']:.3f}  "
                  f"z={pose_goal['z']:.3f}  "
                  f"qx={pose_goal['qx']:.3f}  qy={pose_goal['qy']:.3f}  "
                  f"qz={pose_goal['qz']:.3f}  qw={pose_goal['qw']:.3f}")
            ok += 1
        except (ValueError, RuntimeError) as e:
            print(f"  [WARN] GoalBuilder: {e}")
            fail += 1

    gb.disconnect()

    print(f"\n{SEP}")
    print(f"Resultado: {ok} OK  |  {fail} advertencias")
    print(f"{SEP}\n")


def main():
    parser = argparse.ArgumentParser(description="Prueba PLN → GoalBuilder")
    parser.add_argument("--pln-only", action="store_true",
                        help="Solo PLN, sin conectar a Gazebo")
    parser.add_argument("--model", default="xlmr", choices=["mbert", "xlmr"],
                        help="Modelo PLN a usar (default: xlmr)")
    parser.add_argument("--host", default="localhost",
                        help="IP del rosbridge (default: localhost)")
    parser.add_argument("--port", type=int, default=9090,
                        help="Puerto del rosbridge (default: 9090)")
    parser.add_argument("--texts", nargs="+", default=None,
                        help="Frases a probar (default: batería interna)")
    args = parser.parse_args()

    texts = args.texts if args.texts else DEFAULT_TEXTS

    if args.pln_only:
        run_pln_only(texts, args.model)
    else:
        run_full_pipeline(texts, args.model, args.host, args.port)


if __name__ == "__main__":
    main()
