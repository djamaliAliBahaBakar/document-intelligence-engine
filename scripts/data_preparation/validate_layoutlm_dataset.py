from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk


DATASET_DIR = Path("dataset/processed/layoutlm/dataset")
LABELS_PATH = Path("dataset/processed/layoutlm/labels.json")

# Limite utilisée par LayoutLMv3 lors de l'encodage.
MAX_SEQUENCE_LENGTH = 512


def load_labels() -> tuple[dict[str, int], dict[int, str]]:
    with LABELS_PATH.open(encoding="utf-8") as file:
        labels_config = json.load(file)

    label2id = labels_config["label2id"]

    id2label = {
        int(label_id): label
        for label_id, label in labels_config["id2label"].items()
    }

    return label2id, id2label


def validate_bbox(bbox: list[int]) -> bool:
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, int) for value in bbox)
        and all(0 <= value <= 1000 for value in bbox)
        and bbox[0] <= bbox[2]
        and bbox[1] <= bbox[3]
    )


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATASET_DIR}"
        )

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Vocabulaire introuvable : {LABELS_PATH}"
        )

    dataset = load_from_disk(str(DATASET_DIR))
    label2id, id2label = load_labels()

    o_label_id = label2id["O"]
    valid_label_ids = set(id2label)

    global_counts: Counter[int] = Counter()

    pages_without_annotations: list[str] = []
    pages_over_512_tokens: list[tuple[str, int]] = []
    annotations_after_512: list[tuple[str, str, int]] = []
    errors: list[str] = []

    print("===== VALIDATION DU DATASET LAYOUTLMv3 =====")
    print(f"Nombre de pages : {len(dataset)}")
    print(f"Nombre de labels BIO : {len(label2id)}")
    print()

    for example in dataset:
        page_id = example["id"]
        tokens = example["tokens"]
        bboxes = example["bboxes"]
        ner_tags = example["ner_tags"]
        image_path = Path(example["image_path"])

        print(f"--- {page_id} ---")
        print(f"Tokens : {len(tokens)}")

        # 1. Vérification de l'alignement des listes
        if not (
            len(tokens)
            == len(bboxes)
            == len(ner_tags)
        ):
            errors.append(
                f"{page_id}: longueurs différentes "
                f"tokens={len(tokens)}, "
                f"bboxes={len(bboxes)}, "
                f"ner_tags={len(ner_tags)}"
            )
            print("ERREUR : listes non alignées")
            print()
            continue

        # 2. Vérification de l'image
        if not image_path.exists():
            errors.append(
                f"{page_id}: image introuvable : {image_path}"
            )

        # 3. Vérification des bounding boxes
        invalid_bbox_indexes = [
            index
            for index, bbox in enumerate(bboxes)
            if not validate_bbox(bbox)
        ]

        if invalid_bbox_indexes:
            errors.append(
                f"{page_id}: bounding boxes invalides "
                f"aux index {invalid_bbox_indexes[:10]}"
            )

        # 4. Vérification des identifiants de labels
        unknown_labels = [
            label_id
            for label_id in ner_tags
            if label_id not in valid_label_ids
        ]

        if unknown_labels:
            errors.append(
                f"{page_id}: labels inconnus : "
                f"{sorted(set(unknown_labels))}"
            )

        page_counts = Counter(ner_tags)
        global_counts.update(ner_tags)

        business_label_count = sum(
            count
            for label_id, count in page_counts.items()
            if label_id != o_label_id
        )

        if business_label_count == 0:
            pages_without_annotations.append(page_id)
            print("Annotations métier : aucune")
        else:
            print(
                f"Annotations métier : {business_label_count} token(s)"
            )

        # 5. Vérification de la limite de 512 tokens
        if len(tokens) > MAX_SEQUENCE_LENGTH:
            pages_over_512_tokens.append(
                (page_id, len(tokens))
            )
            print(
                f"ATTENTION : page supérieure à "
                f"{MAX_SEQUENCE_LENGTH} tokens"
            )

        # Une annotation après le token 511 serait perdue
        # avec truncation=True et max_length=512.
        for index, label_id in enumerate(ner_tags):
            if (
                index >= MAX_SEQUENCE_LENGTH
                and label_id != o_label_id
            ):
                annotations_after_512.append(
                    (
                        page_id,
                        id2label[label_id],
                        index,
                    )
                )

        # 6. Affichage des entités annotées
        annotated_tokens = [
            (token, id2label[label_id], index)
            for index, (token, label_id) in enumerate(
                zip(tokens, ner_tags, strict=True)
            )
            if label_id != o_label_id
        ]

        for token, label, index in annotated_tokens:
            print(
                f"  [{index:>4}] "
                f"{token:<30} -> {label}"
            )

        print()

    print("===== DISTRIBUTION GLOBALE =====")

    total_tokens = sum(global_counts.values())

    for label_id, count in global_counts.most_common():
        label = id2label[label_id]
        percentage = (
            count / total_tokens * 100
            if total_tokens
            else 0
        )

        print(
            f"{label:<25} "
            f"{count:>6} "
            f"({percentage:>6.2f} %)"
        )

    print()
    print("===== SYNTHÈSE =====")
    print(f"Nombre total de tokens : {total_tokens}")
    print(
        "Pages avec annotations : "
        f"{len(dataset) - len(pages_without_annotations)}"
    )
    print(
        "Pages sans annotation : "
        f"{len(pages_without_annotations)}"
    )
    print(
        "Pages de plus de 512 tokens : "
        f"{len(pages_over_512_tokens)}"
    )
    print(
        "Annotations situées après le token 511 : "
        f"{len(annotations_after_512)}"
    )
    print(f"Erreurs structurelles : {len(errors)}")

    if pages_without_annotations:
        print("\nPages sans annotation métier :")
        for page_id in pages_without_annotations:
            print(f"  - {page_id}")

    if pages_over_512_tokens:
        print("\nPages supérieures à 512 tokens :")
        for page_id, token_count in pages_over_512_tokens:
            print(f"  - {page_id}: {token_count} tokens")

    if annotations_after_512:
        print("\nAnnotations perdues par troncature :")
        for page_id, label, index in annotations_after_512:
            print(
                f"  - {page_id}: "
                f"{label} au token {index}"
            )

    if errors:
        print("\nErreurs à corriger :")
        for error in errors:
            print(f"  - {error}")

        raise SystemExit(1)

    if global_counts[o_label_id] == 0:
        print("\nERREUR : aucun label O trouvé.")
        raise SystemExit(1)

    print("\nValidation structurelle réussie.")


if __name__ == "__main__":
    main()