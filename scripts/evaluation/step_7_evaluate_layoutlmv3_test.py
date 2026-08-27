import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

import numpy as np
from datasets import Dataset, load_from_disk
from seqeval.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
)


DATASET_DIR = Path(
    "dataset/processed/layoutlm_full/dataset"
)

LABELS_PATH = Path(
    "dataset/processed/layoutlm_full/labels.json"
)


MODEL_DIR = Path(
    "models/layoutlmv3-hyperparameter/lr-5e-05/final"
)

OUTPUT_DIR = Path(
    "models/layoutlmv3-hyperparameter/lr-5e-05/test-evaluation"
)

TEST_METRICS_PATH = OUTPUT_DIR / "test_metrics.json"

MAX_LENGTH = 512

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


def load_labels() -> tuple[dict[str, int], dict[int, str]]:
    with LABELS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
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

    documents: dict[str, list[int]] = defaultdict(list)

    for index, example in enumerate(dataset):
        document_id = example["id"].rsplit(
            "_page_",
            1,
        )[0]

        documents[document_id].append(index)

    document_ids = sorted(documents.keys())
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
        raise ValueError("Fuite entre train et validation.")

    if not train_docs.isdisjoint(test_docs):
        raise ValueError("Fuite entre train et test.")

    if not validation_docs.isdisjoint(test_docs):
        raise ValueError("Fuite entre validation et test.")

    print("\n===== Split final =====")
    print("Documents train      :", len(train_docs))
    print("Documents validation :", len(validation_docs))
    print("Documents test       :", len(test_docs))
    print("Pages test           :", len(test_dataset))

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def encode_example(
    example: dict,
    processor,
) -> dict:
    image = Image.open(
        example["image_path"]
    ).convert("RGB")

    encoding = processor(
        images=image,
        text=example["tokens"],
        boxes=example["bboxes"],
        word_labels=example["ner_tags"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="np",
    )

    return {
        "input_ids": encoding["input_ids"][0],
        "attention_mask": encoding["attention_mask"][0],
        "bbox": encoding["bbox"][0],
        "pixel_values": encoding["pixel_values"][0],
        "labels": encoding["labels"][0],
    }


def encode_dataset(
    dataset: Dataset,
    processor,
) -> Dataset:
    return dataset.map(
        lambda example: encode_example(
            example,
            processor,
        ),
        remove_columns=dataset.column_names,
        desc="Encodage du jeu de test LayoutLMv3",
    )


def build_compute_metrics(
    id2label: dict[int, str],
):
    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction

        predicted_ids = np.argmax(
            logits,
            axis=-1,
        )

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
            "\n===== Distribution réelle sur le test ====="
        )

        for label in id2label.values():
            print(
                f"{label:<25} : "
                f"{real_distribution.get(label, 0)}"
            )

        print(
            "\n===== Distribution prédite sur le test ====="
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

    # 1. Récupérer le modèle final, les labels et le dataset
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    _, id2label = load_labels()

    dataset = load_from_disk(
        str(DATASET_DIR)
    )

    # 2 - Récupérer les dataset de test non utilisés pendant l'entrainement
    _, _, test_raw = split_by_document(
        dataset
    )

    # 3 - Charger 
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        apply_ocr=False,
    )

    model = (
        LayoutLMv3ForTokenClassification.from_pretrained(
            MODEL_DIR
        )
    )

    # Encoder le jeu de test
    test_dataset = encode_dataset(
        test_raw,
        processor,
    )

    evaluation_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_eval_batch_size=2,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=evaluation_args,
        compute_metrics=build_compute_metrics(
            id2label
        ),
    )

    print(
        "\n===== Évaluation finale LayoutLMv3 "
        "sur le test ====="
    )

    # Prédire sans changer les poids - juste de l'inférence
    test_metrics = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test",
    )

    print(
        "\n===== Métriques finales du test ====="
    )

    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, float):
            print(
                f"{metric_name:<30} : "
                f"{metric_value:.4f}"
            )
        else:
            print(
                f"{metric_name:<30} : "
                f"{metric_value}"
            )

    serializable_metrics = {
        key: float(value)
        if isinstance(value, (int, float))
        else value
        for key, value in test_metrics.items()
    }

    with TEST_METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nMétriques sauvegardées dans :",
        TEST_METRICS_PATH,
    )


if __name__ == "__main__":
    main()