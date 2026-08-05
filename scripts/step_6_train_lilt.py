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

MODEL_NAME = "SCUT-DLVCLab/lilt-roberta-en-base"

DATASET_DIR = Path("dataset/processed/layoutlm_full/dataset")
LABELS_PATH = Path("dataset/processed/layoutlm_full/labels.json")

OUTPUT_DIR = Path(
    "models/lilt-photovoltaic-full-split-70-15-15"
)
FINAL_MODEL_DIR = OUTPUT_DIR / "final"
METRICS_CSV_PATH = OUTPUT_DIR / "training_metrics.csv"
FINAL_METRICS_PATH = OUTPUT_DIR / "final_metrics.json"

MAX_LENGTH = 512
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
EPOCH = 10


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


def encode_example(
    example: dict[str, Any],
    tokenizer,
) -> dict[str, Any]:
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


def encode_dataset(
    dataset: Dataset,
    tokenizer,
) -> Dataset:
    return dataset.map(
        lambda example: encode_example(example, tokenizer),
        remove_columns=dataset.column_names,
        desc="Encodage LiLT",
    )


def split_by_document(
    dataset: Dataset,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> tuple[Dataset, Dataset, Dataset]:
    ratio_sum = train_ratio + validation_ratio + test_ratio

    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            "La somme des ratios train, validation et test "
            f"doit être égale à 1.0, reçu : {ratio_sum}"
        )

    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError(
            "Les ratios doivent être strictement positifs."
        )

    documents: dict[str, list[int]] = defaultdict(list)

    for index, example in enumerate(dataset):
        document_id = example["id"].rsplit("_page_", 1)[0]
        documents[document_id].append(index)

    document_ids = sorted(documents.keys())

    if len(document_ids) < 3:
        raise ValueError(
            "Le dataset doit contenir au moins trois documents."
        )

    random.Random(seed).shuffle(document_ids)

    document_count = len(document_ids)
    train_end = int(document_count * train_ratio)
    validation_end = train_end + int(
        document_count * validation_ratio
    )

    train_docs = set(document_ids[:train_end])
    validation_docs = set(
        document_ids[train_end:validation_end]
    )
    test_docs = set(document_ids[validation_end:])

    if not train_docs or not validation_docs or not test_docs:
        raise ValueError(
            "Au moins un des splits train, validation ou test est vide."
        )

    train_indices = [
        index
        for document_id, indices in documents.items()
        if document_id in train_docs
        for index in indices
    ]
    validation_indices = [
        index
        for document_id, indices in documents.items()
        if document_id in validation_docs
        for index in indices
    ]
    test_indices = [
        index
        for document_id, indices in documents.items()
        if document_id in test_docs
        for index in indices
    ]

    train_dataset = dataset.select(train_indices)
    validation_dataset = dataset.select(validation_indices)
    test_dataset = dataset.select(test_indices)

    if not train_docs.isdisjoint(validation_docs):
        raise ValueError("Fuite de documents entre train et validation.")
    if not train_docs.isdisjoint(test_docs):
        raise ValueError("Fuite de documents entre train et test.")
    if not validation_docs.isdisjoint(test_docs):
        raise ValueError("Fuite de documents entre validation et test.")

    if len(train_docs | validation_docs | test_docs) != document_count:
        raise ValueError("Tous les documents n'ont pas été répartis.")

    if (
        len(train_dataset)
        + len(validation_dataset)
        + len(test_dataset)
        != len(dataset)
    ):
        raise ValueError("Toutes les pages n'ont pas été réparties.")

    print("\n===== Split par devis =====")
    print(f"Documents train      : {len(train_docs)}")
    print(f"Documents validation : {len(validation_docs)}")
    print(f"Documents test       : {len(test_docs)}")
    print(f"Pages train          : {len(train_dataset)}")
    print(f"Pages validation     : {len(validation_dataset)}")
    print(f"Pages test           : {len(test_dataset)}")

    return train_dataset, validation_dataset, test_dataset


def print_encoded_label_distribution(
    dataset: Dataset,
    id2label: dict[int, str],
    dataset_name: str,
) -> None:
    label_counts: Counter[int] = Counter()

    for example in dataset:
        label_counts.update(
            int(label_id)
            for label_id in example["labels"]
            if label_id != -100
        )

    print(
        "\n===== Distribution des labels après encodage : "
        f"{dataset_name} ====="
    )
    for label_id, label in id2label.items():
        print(f"{label:<25} : {label_counts.get(label_id, 0)}")


def compute_class_weights(
    dataset: Dataset,
    num_labels: int,
) -> torch.Tensor:
    label_counts: Counter[int] = Counter()

    for example in dataset:
        label_counts.update(
            int(label_id)
            for label_id in example["labels"]
            if label_id != -100
        )

    total_labels = sum(label_counts.values())
    weights = [
        total_labels
        / (num_labels * max(label_counts.get(label_id, 0), 1))
        for label_id in range(num_labels)
    ]

    return torch.tensor(weights, dtype=torch.float)


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


def build_compute_metrics(id2label: dict[int, str]):
    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predicted_ids = np.argmax(logits, axis=-1)

        true_predictions: list[list[str]] = []
        true_labels: list[list[str]] = []

        for prediction_sequence, label_sequence in zip(
            predicted_ids,
            labels,
        ):
            current_predictions: list[str] = []
            current_labels: list[str] = []

            for predicted_id, label_id in zip(
                prediction_sequence,
                label_sequence,
            ):
                if label_id == -100:
                    continue

                current_predictions.append(
                    id2label[int(predicted_id)]
                )
                current_labels.append(
                    id2label[int(label_id)]
                )

            true_predictions.append(current_predictions)
            true_labels.append(current_labels)

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


def build_epoch_metrics(
    log_history: list[dict[str, Any]],
) -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
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
                "epoch": float(entry["epoch"]),
                "train_loss": latest_train_loss,
                "eval_loss": float(entry["eval_loss"]),
                "precision": float(
                    entry.get("eval_precision", 0.0)
                ),
                "recall": float(
                    entry.get("eval_recall", 0.0)
                ),
                "f1": float(entry.get("eval_f1", 0.0)),
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

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    print("\nMétriques sauvegardées dans :", output_path)


def run_smoke_test(
    dataset: Dataset,
    tokenizer,
    model: LiltForTokenClassification,
) -> None:
    example = dataset[0]

    encoding = tokenizer(
        example["tokens"],
        boxes=example["bboxes"],
        word_labels=example["ner_tags"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    print("\n===== Smoke test LiLT =====")
    for key, value in encoding.items():
        print(f"{key:<15} : {value.shape}")

    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)

    print(f"Loss initiale : {outputs.loss.item():.4f}")
    print("Logits        :", outputs.logits.shape)


def main() -> None:
    print("===== Chargement LiLT =====")

    label2id, id2label = load_labels()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = LiltForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    dataset = load_from_disk(str(DATASET_DIR))

    run_smoke_test(dataset, tokenizer, model)

    train_raw, validation_raw, test_raw = split_by_document(
        dataset
    )

    print("\n===== Dataset brut =====")
    print("Train      :", len(train_raw))
    print("Validation :", len(validation_raw))
    print("Test       :", len(test_raw))

    train_dataset = encode_dataset(train_raw, tokenizer)
    validation_dataset = encode_dataset(
        validation_raw,
        tokenizer,
    )
    test_dataset = encode_dataset(test_raw, tokenizer)

    print_encoded_label_distribution(
        train_dataset,
        id2label,
        "train",
    )
    print_encoded_label_distribution(
        validation_dataset,
        id2label,
        "validation",
    )
    print_encoded_label_distribution(
        test_dataset,
        id2label,
        "test",
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
        num_train_epochs=EPOCH,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
        seed=SEED,
        data_seed=SEED,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=build_compute_metrics(id2label),
    )

    print("\n===== Début du fine-tuning LiLT =====")
    train_result = trainer.train()

    epoch_metrics = build_epoch_metrics(
        trainer.state.log_history
    )
    print_metrics_table(epoch_metrics)
    save_metrics_csv(epoch_metrics, METRICS_CSV_PATH)

    print("\n===== Évaluation finale sur la validation =====")
    final_metrics = trainer.evaluate()

    FINAL_METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with FINAL_METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                key: float(value)
                if isinstance(value, (int, float))
                else value
                for key, value in final_metrics.items()
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    for metric_name, metric_value in final_metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name:<30} : {metric_value:.4f}")
        else:
            print(f"{metric_name:<30} : {metric_value}")

    # Le jeu de test reste isolé pendant la sélection du modèle
    # et des hyperparamètres. Il sera évalué une seule fois
    # après le choix définitif de la configuration.

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

    print("\n===== Fine-tuning LiLT terminé =====")
    print(train_result)
    print(f"Modèle sauvegardé dans : {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()
