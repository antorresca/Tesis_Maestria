"""
Evaluación comparativa mBERT vs XLM-RoBERTa sobre el conjunto de test.
Genera tabla de métricas y matriz de confusión.

Uso:
    python src/evaluate.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

INTENTS = ["navigate", "pick", "place", "fetch", "transport", "go_home"]


def load_model(model_key: str):
    model_path = MODELS_DIR / model_key / "best"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def predict_batch(texts, tokenizer, model, max_length=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**encodings).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = np.argmax(probs, axis=-1)
    return preds, probs


def evaluate_model(model_key: str, test_df: pd.DataFrame):
    print(f"\nEvaluando {model_key}...")
    tokenizer, model = load_model(model_key)

    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()

    t0 = time.perf_counter()
    preds, probs = predict_batch(texts, tokenizer, model)
    elapsed = time.perf_counter() - t0

    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average="weighted")
    f1_m = f1_score(labels, preds, average="macro")
    model_size_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / 1e6

    return {
        "model": model_key,
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "inference_time_s": elapsed,
        "ms_per_sample": elapsed / len(texts) * 1000,
        "model_size_mb": model_size_mb,
        "preds": preds,
        "labels": labels,
    }


def main():
    with open(DATA_DIR / "label_map.json") as f:
        label_map = json.load(f)

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Test set: {len(test_df)} ejemplos")

    results = {}
    for model_key in ["mbert", "xlmr"]:
        model_path = MODELS_DIR / model_key / "best"
        if not model_path.exists():
            print(f"  [{model_key}] Modelo no encontrado en {model_path} — saltando")
            continue
        results[model_key] = evaluate_model(model_key, test_df)

    if not results:
        print("No hay modelos entrenados. Corre train.py primero.")
        return

    # --- Tabla comparativa ---
    print("\n" + "=" * 60)
    print("COMPARATIVA mBERT vs XLM-RoBERTa")
    print("=" * 60)
    cols = ["model", "accuracy", "f1_weighted", "f1_macro", "ms_per_sample", "model_size_mb"]
    rows = []
    for r in results.values():
        rows.append({c: r[c] for c in cols})
    summary_df = pd.DataFrame(rows).set_index("model")
    print(summary_df.to_string(float_format="{:.4f}".format))

    # --- Classification report por modelo ---
    for model_key, r in results.items():
        print(f"\n--- Classification Report: {model_key} ---")
        print(classification_report(r["labels"], r["preds"], target_names=INTENTS))

    # --- Confusion matrix ---
    for model_key, r in results.items():
        cm = confusion_matrix(r["labels"], r["preds"])
        cm_df = pd.DataFrame(cm, index=INTENTS, columns=INTENTS)
        print(f"\n--- Confusion Matrix: {model_key} ---")
        print(cm_df.to_string())

    # --- Guardar resultados ---
    out_path = ROOT / "data" / "evaluation_results.json"
    export = {}
    for model_key, r in results.items():
        export[model_key] = {k: v for k, v in r.items() if k not in ("preds", "labels")}
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nResultados guardados en {out_path}")


if __name__ == "__main__":
    main()
