

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_from_disk
from PIL import Image
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "microsoft/layoutlmv3-base"
DATASET_DIR = Path("dataset/processed/layoutlm/dataset")
LABELS_PATH = Path("dataset/processed/layoutlm/labels.json")
OUTPUT_DIR = Path("models/layoutlmv3-photovoltaic")
FINAL_MODEL_DIR = OUTPUT_DIR / "final"
METRICS_CSV_PATH = OUTPUT_DIR / "training_metrics.csv"

MAX_LENGTH = 512
TEST_RATIO = 0.2
SEED = 42


def load_labels() -> tuple[dict[str, int], dict[int, str]]:
    with LABELS_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    label2id = {
        label: int(label_id)
        for label, label_id in payload["label2id"].items()
    }
    id2label = {
        int(label_id): label
        for label_id, label in payload["id2label"].items()
    }
    return label2id, id2label


def encode_example(example: dict[str, Any], processor: LayoutLMv3Processor) -> dict[str, Any]:
    with Image.open(example["image_path"]) as source_image:
        image = source_image.convert("RGB")

    encoding = processor(
        image,
        example["tokens"],
        boxes=example["bboxes"],
        word_labels=example["ner_tags"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "bbox": encoding["bbox"],
        "labels": encoding["labels"],
        "pixel_values": encoding["pixel_values"][0],
    }


def encode_dataset(dataset: Dataset, processor: LayoutLMv3Processor) -> Dataset:
    return dataset.map(
        lambda example: encode_example(example, processor),
        remove_columns=dataset.column_names,
        desc="Encodage",
    )


def split_by_document(
    dataset: Dataset,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> tuple[Dataset, Dataset]:
    documents: dict[str, list[int]] = defaultdict(list)

    for index, example in enumerate(dataset):
        document_id = example["id"].rsplit("_page_", 1)[0]
        documents[document_id].append(index)

    document_ids = list(documents.keys())
    random.Random(seed).shuffle(document_ids)

    split_index = int(len(document_ids) * (1 - test_ratio))
    train_docs = set(document_ids[:split_index])
    eval_docs = set(document_ids[split_index:])

    train_indices = [
        index
        for document_id, indices in documents.items()
        if document_id in train_docs
        for index in indices
    ]
    eval_indices = [
        index
        for document_id, indices in documents.items()
        if document_id in eval_docs
        for index in indices
    ]

    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(eval_indices)

    print("\n===== Split par devis =====")
    print(f"Documents train      : {len(train_docs)}")
    print(f"Documents validation : {len(eval_docs)}")
    print(f"Pages train          : {len(train_dataset)}")
    print(f"Pages validation     : {len(eval_dataset)}")

    print("\nTrain :")
    for document_id in sorted(train_docs):
        print(f"  {document_id}")

    print("\nValidation :")
    for document_id in sorted(eval_docs):
        print(f"  {document_id}")

    return train_dataset, eval_dataset


def print_raw_label_distribution(dataset: Dataset, id2label: dict[int, str]) -> None:
    counts: Counter[int] = Counter()
    for example in dataset:
        counts.update(example["ner_tags"])

    print("\n===== Distribution des labels dans le dataset brut =====")
    for label_id, label in id2label.items():
        print(f"{label:<25} : {counts.get(label_id, 0)}")


def print_encoded_label_distribution(
    dataset: Dataset,
    id2label: dict[int, str],
    dataset_name: str,
) -> None:
    counts: Counter[int] = Counter()

    for example in dataset:
        counts.update(
            int(label_id)
            for label_id in example["labels"]
            if label_id != -100
        )

    print(f"\n===== Distribution des labels après encodage : {dataset_name} =====")
    for label_id, label in id2label.items():
        print(f"{label:<25} : {counts.get(label_id, 0)}")


def compute_class_weights(dataset: Dataset, num_labels: int) -> torch.Tensor:
    counts: Counter[int] = Counter()

    for example in dataset:
        counts.update(
            int(label_id)
            for label_id in example["labels"]
            if label_id != -100
        )

    total = sum(counts.values())
    weights = [
        total / (num_labels * max(counts.get(label_id, 0), 1))
        for label_id in range(num_labels)
    ]
    return torch.tensor(weights, dtype=torch.float)


def build_compute_metrics(id2label: dict[int, str]):
    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predicted_ids = np.argmax(logits, axis=-1)

        true_predictions: list[list[str]] = []
        true_labels: list[list[str]] = []

        for prediction_sequence, label_sequence in zip(predicted_ids, labels):
            current_predictions: list[str] = []
            current_labels: list[str] = []

            for predicted_id, label_id in zip(prediction_sequence, label_sequence):
                if label_id == -100:
                    continue

                current_predictions.append(id2label[int(predicted_id)])
                current_labels.append(id2label[int(label_id)])

            true_predictions.append(current_predictions)
            true_labels.append(current_labels)

        predicted_distribution = Counter(
            label
            for sequence in true_predictions
            for label in sequence
        )
        real_distribution = Counter(
            label
            for sequence in true_labels
            for label in sequence
        )

        print("\n===== Distribution des labels réels en validation =====")
        for label in id2label.values():
            print(f"{label:<25} : {real_distribution.get(label, 0)}")

        print("\n===== Distribution des labels prédits en validation =====")
        for label in id2label.values():
            print(f"{label:<25} : {predicted_distribution.get(label, 0)}")

        return {
            "precision": precision_score(
                true_labels,
                true_predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                true_labels,
                true_predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                true_labels,
                true_predictions,
                zero_division=0,
            ),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    return compute_metrics


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs,
    ):
        model_inputs = inputs.copy()
        labels = model_inputs.pop("labels")
        outputs = model(**model_inputs)
        logits = outputs.logits

        loss_function = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            ignore_index=-100,
        )
        loss = loss_function(
            logits.reshape(-1, model.config.num_labels),
            labels.reshape(-1),
        )

        return (loss, outputs) if return_outputs else loss


def build_epoch_metrics(log_history: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Construit une ligne par évaluation/époque depuis Trainer.state.log_history."""
    rows: list[dict[str, float]] = []
    latest_train_loss: float | None = None
    latest_learning_rate: float | None = None

    for entry in log_history:
        if "loss" in entry:
            latest_train_loss = float(entry["loss"])
        if "learning_rate" in entry:
            latest_learning_rate = float(entry["learning_rate"])

        if "eval_loss" not in entry:
            continue

        rows.append(
            {
                "epoch": float(entry.get("epoch", len(rows) + 1)),
                "train_loss": latest_train_loss if latest_train_loss is not None else float("nan"),
                "eval_loss": float(entry["eval_loss"]),
                "precision": float(entry.get("eval_precision", 0.0)),
                "recall": float(entry.get("eval_recall", 0.0)),
                "f1": float(entry.get("eval_f1", 0.0)),
                "accuracy": float(entry.get("eval_accuracy", 0.0)),
                "learning_rate": (
                    latest_learning_rate
                    if latest_learning_rate is not None
                    else float("nan")
                ),
            }
        )

    return rows


def print_metrics_table(rows: list[dict[str, float]]) -> None:
    if not rows:
        print("\nAucune métrique d'évaluation trouvée.")
        return

    headers = [
        "Epoch",
        "Train loss",
        "Eval loss",
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
        "Learning rate",
    ]
    widths = [7, 12, 11, 11, 9, 9, 10, 14]

    def format_row(values: list[str]) -> str:
        return " | ".join(
            value.rjust(width)
            for value, width in zip(values, widths)
        )

    print("\n===== Tableau récapitulatif des métriques =====")
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))

    for row in rows:
        print(
            format_row(
                [
                    f"{row['epoch']:.0f}",
                    f"{row['train_loss']:.4f}",
                    f"{row['eval_loss']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1']:.4f}",
                    f"{row['accuracy']:.4f}",
                    f"{row['learning_rate']:.2e}",
                ]
            )
        )


def save_metrics_csv(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Métriques sauvegardées dans : {path}")


def inspect_predictions(
    trainer: Trainer,
    dataset: Dataset,
    processor: LayoutLMv3Processor,
    id2label: dict[int, str],
    num_examples: int = 3,
) -> None:
    num_examples = min(num_examples, len(dataset))
    selected_dataset = dataset.select(range(num_examples))
    prediction_output = trainer.predict(selected_dataset)

    predicted_label_ids = np.argmax(prediction_output.predictions, axis=-1)
    true_label_ids = prediction_output.label_ids

    print("\n===== Inspection qualitative =====")

    for example_index in range(num_examples):
        tokens = processor.tokenizer.convert_ids_to_tokens(
            selected_dataset[example_index]["input_ids"]
        )
        predictions = predicted_label_ids[example_index]
        true_labels = true_label_ids[example_index]

        print(f"\n----- Exemple {example_index + 1} -----")
        print(f"{'TOKEN':<30}{'LABEL RÉEL':<25}{'PRÉDICTION':<25}")

        for token, true_id, predicted_id in zip(tokens, true_labels, predictions):
            if true_id == -100:
                continue

            true_label = id2label[int(true_id)]
            predicted_label = id2label[int(predicted_id)]

            if true_label == "O" and predicted_label == "O":
                continue

            marker = "✓" if true_label == predicted_label else "✗"
            print(f"{token:<30}{true_label:<25}{predicted_label:<25}{marker}")


def run_smoke_test(
    dataset: Dataset,
    processor: LayoutLMv3Processor,
    model: LayoutLMv3ForTokenClassification,
) -> None:
    example = dataset[0]

    with Image.open(example["image_path"]) as source_image:
        image = source_image.convert("RGB")

    encoding = processor(
        image,
        example["tokens"],
        boxes=example["bboxes"],
        word_labels=example["ner_tags"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    print("\n===== Smoke test =====")
    for key, value in encoding.items():
        print(key, value.shape)

    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)

    print(f"Loss initiale : {outputs.loss.item():.4f}")
    print("Logits        :", outputs.logits.shape)


def main() -> None:
    dataset = load_from_disk(str(DATASET_DIR))
    label2id, id2label = load_labels()

    print_raw_label_distribution(dataset, id2label)

    processor = LayoutLMv3Processor.from_pretrained(
        MODEL_NAME,
        apply_ocr=False,
    )
    print("\nProcessor chargé avec succès")

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    print("Modèle chargé avec succès")

    run_smoke_test(dataset, processor, model)

    train_raw, eval_raw = split_by_document(dataset)
    train_dataset = encode_dataset(train_raw, processor)
    eval_dataset = encode_dataset(eval_raw, processor)

    print_encoded_label_distribution(train_dataset, id2label, "train")
    print_encoded_label_distribution(eval_dataset, id2label, "validation")

    class_weights = compute_class_weights(train_dataset, len(label2id))

    print("\n===== Poids des classes =====")
    for label_id, weight in enumerate(class_weights):
        print(f"{id2label[label_id]:<25} : {weight.item():.4f}")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )

    print("\n===== Configuration entraînement =====")
    print("Dossier de sortie :", training_args.output_dir)
    print("Learning rate     :", training_args.learning_rate)
    print("Batch train       :", training_args.per_device_train_batch_size)
    print("Nombre d'epochs   :", training_args.num_train_epochs)
    print("Évaluation        :", training_args.eval_strategy)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics(id2label),
    )

    print("\n===== Début du fine-tuning =====")
    train_result = trainer.train()

    metrics_rows = build_epoch_metrics(trainer.state.log_history)
    print_metrics_table(metrics_rows)
    save_metrics_csv(metrics_rows, METRICS_CSV_PATH)

    final_metrics = trainer.evaluate()
    print("\n===== Métriques finales du meilleur checkpoint =====")
    for metric_name in (
        "eval_loss",
        "eval_precision",
        "eval_recall",
        "eval_f1",
        "eval_accuracy",
    ):
        if metric_name in final_metrics:
            print(f"{metric_name:<20} : {final_metrics[metric_name]:.4f}")

    inspect_predictions(
        trainer=trainer,
        dataset=eval_dataset,
        processor=processor,
        id2label=id2label,
        num_examples=3,
    )

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_MODEL_DIR))
    processor.save_pretrained(str(FINAL_MODEL_DIR))

    print("\n===== Fine-tuning terminé =====")
    print(train_result)
    print(f"Modèle sauvegardé dans : {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()
