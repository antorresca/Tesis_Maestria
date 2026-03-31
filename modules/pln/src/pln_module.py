"""
Módulo PLN — Interfaz principal.
Pipeline de dos etapas:
  1. Clasificación de intención (mBERT o XLM-RoBERTa fine-tuneado)
  2. Extracción de entidades (reglas)

Uso como librería:
    from pln_module import PLNModule
    pln = PLNModule(model_type="xlmr")
    result = pln.predict("lleva el cubo azul a la mesa")
    # → {"intent": "place", "target": "cubo azul", "destination": "mesa", "confidence": 0.97}

Uso como script:
    python src/pln_module.py --model xlmr
    python src/pln_module.py --model xlmr --text "go to the kitchen"
"""

import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from entity_extractor import extract

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"


class PLNModule:
    def __init__(self, model_type: str = "xlmr", model_path: Optional[str] = None):
        """
        Args:
            model_type: "mbert" o "xlmr"
            model_path: ruta personalizada al modelo (opcional)
        """
        if model_path is None:
            model_path = str(MODELS_DIR / model_type / "best")

        # Leer label_map del config.json del modelo (siempre sincronizado con el modelo cargado)
        with open(Path(model_path) / "config.json") as f:
            cfg = json.load(f)
        self._label_map: dict = cfg["id2label"]  # {"0": "navigate", ...}

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        self.model_type = model_type
        print(f"[PLN] Modelo '{model_type}' cargado desde {model_path} ({self._device})")

    def predict(self, text: str) -> dict:
        """
        Clasifica la intención y extrae entidades del texto de entrada.

        Returns:
            dict con claves: intent, target, destination, confidence
        """
        encoding = self._tokenizer(
            text,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**encoding).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_id = int(probs.argmax())
        confidence = float(probs[pred_id])
        intent = self._label_map[str(pred_id)]

        entities = extract(intent, text)

        return {
            "intent": intent,
            "target": entities.target,
            "destination": entities.destination,
            "confidence": round(confidence, 4),
        }


# --- CLI interactiva ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Módulo PLN interactivo")
    parser.add_argument("--model", choices=["mbert", "xlmr"], default="xlmr")
    parser.add_argument("--text", type=str, default=None, help="Texto a clasificar (opcional)")
    args = parser.parse_args()

    pln = PLNModule(model_type=args.model)

    if args.text:
        result = pln.predict(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("\nModo interactivo — escribe un comando (Ctrl+C para salir):\n")
        while True:
            try:
                text = input(">> ").strip()
                if not text:
                    continue
                result = pln.predict(text)
                print(json.dumps(result, ensure_ascii=False, indent=2))
            except KeyboardInterrupt:
                print("\nSaliendo.")
                break
