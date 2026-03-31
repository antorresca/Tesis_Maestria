"""
Mezcla seed_dataset.csv + augmented_dataset.csv, deduplica y genera splits train/val/test.
Uso: python src/prepare_dataset.py
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

INTENTS = ["navigate", "pick", "place"]

# Remapeo de 6 clases originales → 3 clases atómicas
# go_home es un navigate con destination="home" (entity extractor lo resuelve)
# fetch es un pick (Task Planning descompone en move_to + grasp)
# transport es un place (Task Planning descompone en move_to + grasp + move_to + release)
INTENT_REMAP = {
    "go_home": "navigate",
    "fetch": "pick",
    "transport": "place",
}

DATA_DIR = Path(__file__).parent.parent / "data"

def load_data() -> pd.DataFrame:
    frames = []
    seed_path = DATA_DIR / "raw" / "seed_dataset.csv"
    if seed_path.exists():
        frames.append(pd.read_csv(seed_path))
    aug_path = DATA_DIR / "raw" / "augmented_dataset.csv"
    if aug_path.exists():
        frames.append(pd.read_csv(aug_path))
    if not frames:
        raise FileNotFoundError("No se encontró ningún CSV en data/raw/")
    df = pd.concat(frames, ignore_index=True)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text"].str.strip().str.lower()
    df = df.drop_duplicates(subset=["text"])
    df = df[df["intent"].isin(INTENTS)]
    df = df.dropna(subset=["text", "intent"])
    return df.reset_index(drop=True)

def main():
    df = load_data()
    df = clean(df)

    print(f"Total ejemplos: {len(df)}")
    print(df.groupby(["intent", "lang"]).size().to_string())

    label2id = {intent: i for i, intent in enumerate(INTENTS)}
    df["label"] = df["intent"].map(label2id)

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["intent"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["intent"], random_state=42)

    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(f"\nSplits guardados en {out_dir}")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    label_map = {str(v): k for k, v in label2id.items()}
    import json
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print("  label_map.json guardado")

if __name__ == "__main__":
    main()
