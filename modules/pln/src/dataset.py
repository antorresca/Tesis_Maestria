"""
PyTorch Dataset para clasificación de intenciones.
"""

import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def _augment_text(text: str, word_del_p: float = 0.05, char_swap_p: float = 0.02) -> str:
    """Aplica ruido al texto para regularización durante entrenamiento.

    - word_del_p: probabilidad de eliminar cada palabra (nunca elimina todas).
    - char_swap_p: probabilidad de intercambiar un carácter con el siguiente.
    """
    # Random word deletion
    words = text.split()
    if len(words) > 1:
        words = [w for w in words if random.random() > word_del_p] or words[:1]

    # Character-level swap (simula errores tipográficos)
    result = []
    for word in words:
        chars = list(word)
        for i in range(len(chars) - 1):
            if random.random() < char_swap_p:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
        result.append("".join(chars))

    return " ".join(result)


class IntentDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        augment: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["text"])
        label = int(self.df.iloc[idx]["label"])

        if self.augment:
            text = _augment_text(text)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
