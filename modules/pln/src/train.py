"""
Fine-tuning de mBERT o XLM-RoBERTa para clasificación de intenciones.

Uso:
    python src/train.py --model mbert --epochs 10 --batch_size 16
    python src/train.py --model xlmr  --epochs 10 --batch_size 16

Modelos:
    mbert  → bert-base-multilingual-cased
    xlmr   → xlm-roberta-base
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from dataset import IntentDataset

MODEL_IDS = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
}

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main(args):
    model_name = MODEL_IDS[args.model]
    print(f"Modelo: {model_name}")

    with open(DATA_DIR / "label_map.json") as f:
        label_map = json.load(f)
    num_labels = len(label_map)
    id2label = label_map
    label2id = {v: int(k) for k, v in label_map.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = IntentDataset(DATA_DIR / "train.csv", tokenizer)
    val_dataset = IntentDataset(DATA_DIR / "val.csv", tokenizer)

    output_dir = MODELS_DIR / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="best",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Entrenando...")
    trainer.train()

    print(f"Modelo guardado en {output_dir}")
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))

    val_results = trainer.evaluate()
    print(f"\n=== Resultados de Validación ({args.model}) ===")
    for k, v in val_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mbert", "xlmr"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
