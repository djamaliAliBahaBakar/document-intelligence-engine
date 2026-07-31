from pathlib import Path
import json

from datasets import load_from_disk
from PIL import Image
import torch
import torch.nn as nn

from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)

from transformers import TrainingArguments
from transformers import Trainer
from collections import Counter, defaultdict

from datasets import Dataset
from collections import Counter

import numpy as np
from seqeval.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import numpy as np



DATASET_DIR = Path("dataset/processed/layoutlm/dataset")
LABELS_PATH = Path("dataset/processed/layoutlm/labels.json")

def encode_example(example, processor):
    image = Image.open(example["image_path"]).convert("RGB")

    encoding = processor(
        image,
        example["tokens"],
        boxes=example["bboxes"],
        word_labels=example["ner_tags"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "bbox": encoding["bbox"],
        "labels": encoding["labels"],
        "pixel_values": encoding["pixel_values"][0],
    }

def contains_business_annotation(
    example: dict,
    o_label_id: int,
) -> bool:
    return any(
        label_id != o_label_id
        for label_id in example["ner_tags"]
    )



def split_by_document(
    dataset: Dataset,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    import random

    random.seed(seed)

    documents = defaultdict(list)

    for index, example in enumerate(dataset):
        document_id = example["id"].rsplit("_page_", 1)[0]
        documents[document_id].append(index)

    document_ids = list(documents.keys())
    random.shuffle(document_ids)

    split_index = int(len(document_ids) * (1 - test_ratio))

    train_docs = set(document_ids[:split_index])
    test_docs = set(document_ids[split_index:])

    train_indices = []
    test_indices = []

    for document_id, indices in documents.items():
        if document_id in train_docs:
            train_indices.extend(indices)
        else:
            test_indices.extend(indices)

    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)

    print("\n===== Split par devis =====")
    print(f"Documents train      : {len(train_docs)}")
    print(f"Documents validation : {len(test_docs)}")
    print(f"Pages train          : {len(train_dataset)}")
    print(f"Pages validation     : {len(test_dataset)}")

    print("\nTrain :")
    for doc in sorted(train_docs):
        print(" ", doc)

    print("\nValidation :")
    for doc in sorted(test_docs):
        print(" ", doc)

    return train_dataset, test_dataset


def encode_dataset(
    dataset,
    processor,
):
    return dataset.map(
        lambda example: encode_example(
            example,
            processor,
        ),
        remove_columns=dataset.column_names,
        desc="Encodage",
    )


def build_compute_metrics(id2label: dict[int, str]):
    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction

        predictions = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):
            current_predictions = []
            current_labels = []

            for predicted_id, label_id in zip(prediction, label):
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
            count = real_distribution.get(label, 0)
            print(f"{label:<25} : {count}")

        print("\n===== Distribution des labels prédits en validation =====")
        for label in id2label.values():
            count = predicted_distribution.get(label, 0)
            print(f"{label:<25} : {count}")

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

def encode_dataset(
    dataset,
    processor,
):
    return dataset.map(
        lambda example: encode_example(
            example,
            processor,
        ),
        remove_columns=dataset.column_names,
        desc="Encodage",
    )


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

        
        inputs = inputs.copy()

        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            ignore_index=-100,
        )

        loss = loss_fct(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )

        if return_outputs:
            return loss, outputs

        return loss

import numpy as np


def inspect_predictions(
    trainer,
    dataset,
    processor,
    id2label,
    num_examples: int = 3,
) -> None:
    num_examples = min(num_examples, len(dataset))

    selected_dataset = dataset.select(range(num_examples))

    prediction_output = trainer.predict(selected_dataset)

    predicted_label_ids = np.argmax(
        prediction_output.predictions,
        axis=-1,
    )

    true_label_ids = prediction_output.label_ids

    print("\n===== Inspection qualitative =====")

    for example_index in range(num_examples):
        input_ids = selected_dataset[example_index]["input_ids"]

        tokens = processor.tokenizer.convert_ids_to_tokens(
            input_ids
        )

        predictions = predicted_label_ids[example_index]
        true_labels = true_label_ids[example_index]

        print(
            f"\n----- Exemple {example_index + 1} -----"
        )

        print(
            f"{'TOKEN':<30}"
            f"{'LABEL RÉEL':<25}"
            f"{'PRÉDICTION':<25}"
        )

        for token, true_id, predicted_id in zip(
            tokens,
            true_labels,
            predictions,
        ):
            # Tokens spéciaux ou sous-tokens ignorés par la loss
            if true_id == -100:
                continue

            true_label = id2label[int(true_id)]
            predicted_label = id2label[int(predicted_id)]

            # Pour éviter d'afficher les centaines de tokens O/O
            if true_label == "O" and predicted_label == "O":
                continue

            marker = (
                "✓"
                if true_label == predicted_label
                else "✗"
            )

            print(
                f"{token:<30}"
                f"{true_label:<25}"
                f"{predicted_label:<25}"
                f"{marker}"
            )

def main() -> None:
    dataset = load_from_disk(str(DATASET_DIR))


    print("\n===== Exemple OCR brut =====")

    example = dataset[0]

    print("\n===== Diagnostic Processor =====")

    image = Image.open(example["image_path"]).convert("RGB")

    processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False,
    )
    print("\nProcessor chargé avec succès")

    encoding = processor(
        image,
        example["tokens"],
        boxes=example["bboxes"],
        truncation=True,
        padding=False,
    )

    input_ids = encoding["input_ids"]

    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)

    print("\n===== Vérification du décodage =====")

    decoded = processor.tokenizer.decode(input_ids)

    print(decoded[:500])

    print("\n===== Tokens suspects =====")

    for token in tokens:
        if any(c in token for c in ["â", "Ã", "Ĥ"]):
            print(token)

    word_ids = encoding.word_ids()

    print("\n===== Correspondance OCR -> Tokens =====")

    for token, word_id in zip(tokens, word_ids):
        if word_id is None:
            continue

        if any(c in token for c in ["â", "Ã", "Ĥ"]):
            print(
                f"OCR : {example['tokens'][word_id]!r} "
                f"--> Token : {token!r}"
            )

    print("ID :", example["id"])
    print()

    print("Premiers mots OCR :")
    for word in example["tokens"][:50]:
        print(repr(word))


    with LABELS_PATH.open("r", encoding="utf-8") as file:
        labels_payload = json.load(file)

    label2id = {
        label: int(label_id)
        for label, label_id in labels_payload["label2id"].items()
    }

    id2label = {
        int(label_id): label
        for label_id, label in labels_payload["id2label"].items()
    }

    compute_metrics = build_compute_metrics(id2label)

    raw_label_counts = Counter()

    for example in dataset:
        raw_label_counts.update(example["ner_tags"])

    print("\n===== Distribution des labels dans le dataset brut =====")

    for label_id, count in sorted(raw_label_counts.items()):
        print(f"{id2label[label_id]:20} : {count}")

    processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False,
    )
    print("\nProcessor chargé avec succès")


   

    model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
    )

    print("Modèle chargé avec succès")
    example = dataset[0]

    image = Image.open(example["image_path"]).convert("RGB")

    encoding = processor(
        image,
        example["tokens"],
        boxes=example["bboxes"],
        word_labels=example["ner_tags"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    print("\nEncodage réussi")
    for key, value in encoding.items():
        print(key, value.shape)


    model.eval()

    with torch.no_grad():
        outputs = model(**encoding)

    print("\nForward pass réussi")
    print("Loss :", outputs.loss.item())
    print("Logits :", outputs.logits.shape)

    #
    # Split AVANT encodage
    #
    train_dataset_raw, eval_dataset_raw = split_by_document(
        dataset,
        test_ratio=0.2,
        seed=42,
    )

    #
    # Encodage indépendant
    #
    train_dataset = encode_dataset(
        train_dataset_raw,
        processor,
    )

    eval_dataset = encode_dataset(
        eval_dataset_raw,
        processor,
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

    print()

    print("===== Dataset =====")
    print("Train :", len(train_dataset))
    print("Validation :", len(eval_dataset))
    
    training_args = TrainingArguments(
        output_dir="models/layoutlmv3-photovoltaic",
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
    )

    print("\n===== Configuration entraînement =====")
    print("Dossier de sortie :", training_args.output_dir)
    print("Learning rate :", training_args.learning_rate)
    print("Batch train :", training_args.per_device_train_batch_size)
    print("Nombre d'epochs :", training_args.num_train_epochs)
    print("Évaluation :", training_args.eval_strategy)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    print("\n===== Trainer =====")
    print("Trainer créé avec succès")
    print("Train dataset :", len(trainer.train_dataset))
    print("Eval dataset :", len(trainer.eval_dataset))
    print("\nDataset brut")

    print("\n===== Début du fine-tuning =====")

    train_result = trainer.train()

    inspect_predictions(
        trainer=trainer,
        dataset=eval_dataset,
        processor=processor,
        id2label=id2label,
        num_examples=3,
    )

    print("\n===== Fine-tuning terminé =====")
    print(train_result)

    model_output_dir = "models/layoutlmv3-photovoltaic/final"

    trainer.save_model(model_output_dir)
    processor.save_pretrained(model_output_dir)

    print(f"\nModèle sauvegardé dans : {model_output_dir}")
    #print(dataset)
    #print(dataset.features)

    #print(f"Nombre d'exemples : {len(dataset)}")
    #print(f"Nombre de labels BIO : {len(label2id)}")

    #print("label2id :", label2id)
    #print("id2label :", id2label)



if __name__ == "__main__":
    main()