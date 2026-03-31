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
    EarlyStoppingCallback,
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

    # Congelar capas bajas del encoder para reducir overfitting en datasets pequeños.
    # Solo se entrenan las últimas `args.unfreeze_layers` capas + el clasificador.
    if args.unfreeze_layers > 0:
        for name, param in model.base_model.named_parameters():
            param.requires_grad = False
        # Descongelar las últimas N capas del encoder
        encoder_layers = model.base_model.encoder.layer
        for layer in encoder_layers[-args.unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        # El clasificador siempre se entrena
        for param in model.classifier.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Parámetros entrenables: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    train_dataset = IntentDataset(DATA_DIR / "train.csv", tokenizer, max_length=args.max_length, augment=args.augment)
    val_dataset = IntentDataset(DATA_DIR / "val.csv", tokenizer, max_length=args.max_length)

    output_dir = MODELS_DIR / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-5,
        warmup_steps=50,
        weight_decay=args.weight_decay,
        label_smoothing_factor=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_f1_weighted",
        greater_is_better=True,
        logging_steps=10,
        fp16=torch.cuda.is_available() and not args.cpu,
        gradient_checkpointing=not args.cpu,
        optim="adamw_bnb_8bit" if not args.cpu else "adamw_torch",
        use_cpu=args.cpu,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={"train": train_dataset, "val": val_dataset},
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    print("Entrenando...")
    trainer.train()

    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))
    print(f"Modelo guardado en {output_dir / 'best'}")

    # Guardar métricas por época (train + val juntos)
    # Con eval_dataset dict, los entries tienen prefijo "eval_train_*" y "eval_val_*"
    epochs_metrics = []
    train_eval_buffer = {}
    for entry in trainer.state.log_history:
        if "eval_train_loss" in entry:
            train_eval_buffer = entry
        elif "eval_val_loss" in entry:
            ep = int(entry["epoch"])
            epochs_metrics.append({
                "epoch": ep,
                "train_loss":     round(train_eval_buffer.get("eval_train_loss", float("nan")), 4),
                "train_accuracy": round(train_eval_buffer.get("eval_train_accuracy", float("nan")), 4),
                "train_f1_weighted": round(train_eval_buffer.get("eval_train_f1_weighted", float("nan")), 4),
                "train_f1_macro":    round(train_eval_buffer.get("eval_train_f1_macro", float("nan")), 4),
                "val_loss":       round(entry["eval_val_loss"], 4),
                "val_accuracy":   round(entry["eval_val_accuracy"], 4),
                "val_f1_weighted": round(entry["eval_val_f1_weighted"], 4),
                "val_f1_macro":    round(entry["eval_val_f1_macro"], 4),
            })
            train_eval_buffer = {}

    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(epochs_metrics, f, indent=2)
    print(f"Métricas por época guardadas en {metrics_path}")

    print(f"\n=== Resultados finales ({args.model}) ===")
    for row in epochs_metrics[-1:]:
        for k, v in row.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mbert", "xlmr"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--cpu", action="store_true", help="Forzar CPU (fallback si no hay VRAM suficiente)")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Regularización L2 (default 0.1 para reducir overfitting)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping: épocas sin mejora en val antes de detener")
    parser.add_argument("--unfreeze_layers", type=int, default=4, help="Número de capas del encoder a entrenar (0=todas). Default 4 para datasets pequeños.")
    parser.add_argument("--augment", action="store_true", help="Activar augmentación online (word deletion + char swap) solo en train.")
    args = parser.parse_args()
    main(args)
