from pathlib import Path

import torch
import json

from transformers import (
    AutoTokenizer,
    LiltForTokenClassification,
)

from datasets import load_from_disk

MODEL_NAME = "SCUT-DLVCLab/lilt-roberta-en-base"

LABELS_PATH = Path(
    "dataset/processed/layoutlm/labels.json"
)

DATASET_DIR = Path(
    "dataset/processed/layoutlm/dataset"
)

MAX_LENGTH = 512


def load_labels() -> tuple[
    dict[str, int],
    dict[int, str],
]:
    with LABELS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    label2id = {
        label: int(label_id)
        for label, label_id
        in payload["label2id"].items()
    }

    id2label = {
        int(label_id): label
        for label_id, label
        in payload["id2label"].items()
    }

    return label2id, id2label

def encode_example(
    example: dict,
    tokenizer,
) -> dict:
    encoding = tokenizer(
        example["tokens"],
        boxes=example["bboxes"],
        word_labels=example["ner_tags"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "bbox": encoding["bbox"],
        "labels": encoding["labels"],
    }

def main() -> None:
    print("===== Chargement LiLT =====")

    label2id, id2label = load_labels()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = LiltForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    print(
        "Tokenizer :",
        tokenizer.__class__.__name__,
    )

    print(
        "Model :",
        model.__class__.__name__,
    )

    print(
        "Type :",
        model.config.model_type,
    )

    print(
        "Labels :",
        model.config.num_labels,
    )

    print(
        "id2label :",
        model.config.id2label,
    )

    dataset = load_from_disk(
        str(DATASET_DIR)
    )

    example = dataset[0]

    print("\n===== Exemple sélectionné =====")
    print("ID :", example["id"])
    print("Nombre de mots :", len(example["tokens"]))
    print("Nombre de bboxes :", len(example["bboxes"]))
    print("Nombre de labels :", len(example["ner_tags"]))

    encoding = encode_example(
        example,
        tokenizer,
    )

    print("\n===== Encodage LiLT =====")

    for key, value in encoding.items():
        print(
            f"{key:<15} : {value.shape}"
    )

    valid_labels = [
        int(label_id)
        for label_id in encoding["labels"][0]
        if int(label_id) != -100
    ]

    print(
        "Nombre de labels évalués :",
        len(valid_labels),
    )

    print("\n===== Labels métier après encodage =====")

    tokens = tokenizer.convert_ids_to_tokens(
        encoding["input_ids"][0]
    )

    for token, label_id in zip(
        tokens,
        encoding["labels"][0],
    ):
        label_id = int(label_id)

        if label_id == -100:
            continue

        label = id2label[label_id]

        if label == "O":
            continue

        print(
            f"{token:<30} -> {label}"
        )

    print("\n===== Forward pass LiLT =====")

    model.eval()

    with torch.no_grad():
        outputs = model(**encoding)

    print("Loss :", outputs.loss.item())
    print("Logits :", outputs.logits.shape)



if __name__ == "__main__":
    main()