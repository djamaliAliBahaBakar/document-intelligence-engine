from pathlib import Path

import torch.nn as nn

from transformers import Trainer
import csv
from typing import Any

import torch
import json
import random

import numpy as np

from seqeval.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from transformers import (
    AutoTokenizer,
    LiltForTokenClassification,
    Trainer,
    TrainingArguments,
)
from collections import Counter, defaultdict

from datasets import Dataset, load_from_disk

MODEL_NAME = "SCUT-DLVCLab/lilt-roberta-en-base"

LABELS_PATH = Path(
    "dataset/processed/layoutlm/labels.json"
)

DATASET_DIR = Path(
    "dataset/processed/layoutlm/dataset"
)

OUTPUT_DIR = Path(
    "models/lilt-photovoltaic"
)

FINAL_MODEL_DIR = OUTPUT_DIR / "final"

METRICS_CSV_PATH = (
    OUTPUT_DIR / "training_metrics.csv"
)

FINAL_METRICS_PATH = (
    OUTPUT_DIR / "final_metrics.json"
)


MAX_LENGTH = 512
TEST_RATIO = 0.2
SEED = 42


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

"""Transforme une page brute en entrée du modèle"""
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
    )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "bbox": encoding["bbox"],
        "labels": encoding["labels"],
    }

"""Applique encode_example à toutes les pages du dataset"""
def encode_dataset(
    dataset,
    tokenizer,
):
    return dataset.map(
        lambda example: encode_example(
            example,
            tokenizer,
        ),
        remove_columns=dataset.column_names,
        desc="Encodage LiLT",
    )

"""Regroupe les pages par devis puis separe les devis en 2 blocs : train et validate
    Le pourcendate de validation est défini en entrée"""
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

def print_encoded_label_distribution(
    dataset,
    id2label: dict[int, str],
    dataset_name: str,
) -> None:
    label_counts = Counter()

    for example in dataset:
        for label_id in example["labels"]:
            if label_id == -100:
                continue

            label_counts[int(label_id)] += 1

    print(
        f"\n===== Distribution des labels après encodage : "
        f"{dataset_name} ====="
    )

    for label_id, label in id2label.items():
        print(
            f"{label:<25} : "
            f"{label_counts.get(label_id, 0)}"
        )

def build_epoch_metrics(
    log_history: list[dict[str, Any]],
) -> list[dict[str, float | None]]:
    rows = []

    latest_train_loss = None
    latest_learning_rate = None

    for entry in log_history:
        if "loss" in entry:
            latest_train_loss = float(entry["loss"])

        if "learning_rate" in entry:
            latest_learning_rate = float(
                entry["learning_rate"]
            )

        if "eval_loss" not in entry:
            continue

        rows.append(
            {
                "epoch": float(entry["epoch"]),
                "train_loss": latest_train_loss,
                "eval_loss": float(entry["eval_loss"]),
                "precision": float(
                    entry.get("eval_precision", 0.0)
                ),
                "recall": float(
                    entry.get("eval_recall", 0.0)
                ),
                "f1": float(
                    entry.get("eval_f1", 0.0)
                ),
                "accuracy": float(
                    entry.get("eval_accuracy", 0.0)
                ),
                "learning_rate": latest_learning_rate,
            }
        )

    return rows

def print_metrics_table(
    rows: list[dict[str, float | None]],
) -> None:
    if not rows:
        print("\nAucune métrique par époque disponible.")
        return

    print("\n===== Métriques LiLT par époque =====")

    header = (
        f"{'Epoch':>7} | "
        f"{'Train loss':>12} | "
        f"{'Eval loss':>11} | "
        f"{'Precision':>10} | "
        f"{'Recall':>8} | "
        f"{'F1':>8} | "
        f"{'Accuracy':>10} | "
        f"{'Learning rate':>13}"
    )

    print(header)
    print("-" * len(header))

    for row in rows:
        train_loss = row["train_loss"]
        learning_rate = row["learning_rate"]

        print(
            f"{row['epoch']:>7.0f} | "
            f"{train_loss if train_loss is not None else 0:>12.4f} | "
            f"{row['eval_loss']:>11.4f} | "
            f"{row['precision']:>10.4f} | "
            f"{row['recall']:>8.4f} | "
            f"{row['f1']:>8.4f} | "
            f"{row['accuracy']:>10.4f} | "
            f"{learning_rate if learning_rate is not None else 0:>13.2e}"
        )

def save_metrics_csv(
    rows: list[dict[str, float | None]],
    output_path: Path,
) -> None:
    if not rows:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "epoch",
        "train_loss",
        "eval_loss",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "learning_rate",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "\nMétriques sauvegardées dans :",
        output_path,
    )

def run_smoke_test(
    dataset,
    tokenizer,
    model,
):
    example = dataset[0]

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

    model.eval()

    with torch.no_grad():
        outputs = model(**encoding)

    print("Loss :", outputs.loss.item())
    print("Logits :", outputs.logits.shape)

"""Calcul les poids des labels. Les labels sureprésentés auront un poids plus faible et inversement"""
def compute_class_weights(
    dataset,
    num_labels: int,
) -> torch.Tensor:
    label_counts = Counter()

    for example in dataset:
        for label_id in example["labels"]:
            if label_id == -100:
                continue

            label_counts[int(label_id)] += 1

    total_labels = sum(label_counts.values())

    class_weights = []

    for label_id in range(num_labels):
        count = label_counts.get(label_id, 1)

        weight = total_labels / (
            num_labels * count
        )

        class_weights.append(weight)

    return torch.tensor(
        class_weights,
        dtype=torch.float,
    )

class WeightedTrainer(Trainer):
    def __init__(
        self,
        class_weights: torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
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
            logits.reshape(
                -1,
                model.config.num_labels,
            ),
            labels.reshape(-1),
        )

        if return_outputs:
            return loss, outputs

        return loss

def build_compute_metrics(
    id2label: dict[int, str],
):
    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction

        predicted_ids = np.argmax(
            logits,
            axis=-1,
        )

        true_predictions = []
        true_labels = []

        for prediction, label in zip(
            predicted_ids,
            labels,
        ):
            current_predictions = []
            current_labels = []

            for predicted_id, label_id in zip(
                prediction,
                label,
            ):
                if label_id == -100:
                    continue

                current_predictions.append(
                    id2label[int(predicted_id)]
                )

                current_labels.append(
                    id2label[int(label_id)]
                )

            true_predictions.append(
                current_predictions
            )

            true_labels.append(
                current_labels
            )

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

        print(
            "\n===== Distribution des labels réels "
            "en validation ====="
        )

        for label in id2label.values():
            print(
                f"{label:<25} : "
                f"{real_distribution.get(label, 0)}"
            )

        print(
            "\n===== Distribution des labels prédits "
            "en validation ====="
        )

        for label in id2label.values():
            print(
                f"{label:<25} : "
                f"{predicted_distribution.get(label, 0)}"
            )

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
            "accuracy": accuracy_score(
                true_labels,
                true_predictions,
            ),
        }

    return compute_metrics

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

    dataset = load_from_disk(
        str(DATASET_DIR)
    )
    train_dataset_raw, eval_dataset_raw = split_by_document(
        dataset,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    print("\n===== Dataset brut =====")
    print("Train :", len(train_dataset_raw))
    print("Validation :", len(eval_dataset_raw))

    train_dataset = encode_dataset(
        train_dataset_raw,
        tokenizer,
    )

    eval_dataset = encode_dataset(
        eval_dataset_raw,
        tokenizer,
    )

    print_encoded_label_distribution(
        train_dataset,
        id2label,
        "train",
    )

    print_encoded_label_distribution(
        eval_dataset,
        id2label,
        "validation",
    )

    class_weights = compute_class_weights(
        train_dataset,
        num_labels=len(label2id),
    )

    print("\n===== Poids des classes =====")

    for label_id, weight in enumerate(class_weights):
        print(
            f"{id2label[label_id]:<25} : "
            f"{weight.item():.4f}"
        )
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
        report_to="none",
        save_total_limit=2,
        seed=SEED,
        data_seed=SEED,
    )

    compute_metrics = build_compute_metrics(
        id2label
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n===== Trainer LiLT =====")
    print("Trainer créé avec succès")
    print("Train dataset :", len(trainer.train_dataset))
    print("Eval dataset :", len(trainer.eval_dataset))
    print("Modèle :", trainer.model.__class__.__name__)

    print("\n===== Configuration Trainer =====")

    print(
        "Compute metrics :",
        trainer.compute_metrics is not None,
    )

    print(
        "Nombre de labels :",
        trainer.model.config.num_labels,
    )

    print(
        "Best metric :",
        trainer.args.metric_for_best_model,
    )

    print(
        "Learning rate :",
        trainer.args.learning_rate,
    )

    print(
        "Epochs :",
        trainer.args.num_train_epochs,
    )

    print("\n===== Début du fine-tuning LiLT =====")

    train_result = trainer.train()

    epoch_metrics = build_epoch_metrics(
    trainer.state.log_history
)

    print_metrics_table(epoch_metrics)

    save_metrics_csv(
        epoch_metrics,
        METRICS_CSV_PATH,
    )

    print("\n===== Fine-tuning LiLT terminé =====")
    print(train_result)

    print("\n===== Évaluation finale du meilleur checkpoint =====")

    final_metrics = trainer.evaluate()

    print("\n===== Sauvegarde du modèle LiLT =====")

    trainer.save_model(
        str(FINAL_MODEL_DIR)
    )

    tokenizer.save_pretrained(
        str(FINAL_MODEL_DIR)
    )

    print(
        "Modèle et tokenizer sauvegardés dans :",
        FINAL_MODEL_DIR,
    )

    for metric_name, metric_value in final_metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name:<30} : {metric_value:.4f}")
        else:
            print(f"{metric_name:<30} : {metric_value}")


if __name__ == "__main__":
    main()