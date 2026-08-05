from pathlib import Path
import json

from datasets import Dataset
from collections import Counter

from scripts.utils.labels import (
    build_label_vocab,
    encode_labels,
    extract_entity_labels,
)
from scripts.utils.labelstudio import (
    build_layoutlm_example_from_task,
    extract_annotations,
    extract_document_id,
    extract_image_name,
    load_export,
)


EXPORT_PATH = Path(
    "label_studio_data/export/annotations_all.json"
)

OCR_DIR = Path("dataset/ocr")

OUTPUT_DIR = Path("dataset/processed/layoutlm_full")
DATASET_DIR = OUTPUT_DIR / "dataset"
LABELS_PATH = OUTPUT_DIR / "labels.json"


def main() -> None:
    tasks = load_export(EXPORT_PATH)

    total_pages = len(tasks)
    annotated_tasks: list[dict] = []
    skipped_pages: list[str] = []

    all_documents: set[str] = set()
    kept_documents: set[str] = set()

    for task in tasks:
        image_name = extract_image_name(task)
        document_id = extract_document_id(image_name)

        all_documents.add(document_id)

        annotations = extract_annotations(task)

        if not annotations:
            skipped_pages.append(image_name)
            continue

        annotated_tasks.append(task)
        kept_documents.add(document_id)

    excluded_documents = sorted(
        all_documents - kept_documents
    )

    print("\n===== Filtrage des pages =====")
    print(f"Pages totales       : {total_pages}")
    print(f"Pages conservées    : {len(annotated_tasks)}")
    print(f"Pages ignorées      : {len(skipped_pages)}")
    print(f"Documents totaux    : {len(all_documents)}")
    print(f"Documents conservés : {len(kept_documents)}")
    print(f"Documents exclus    : {len(excluded_documents)}")

    if excluded_documents:
        print("\nDocuments exclus :")
        for document_id in excluded_documents:
            print(f"  - {document_id}")

    if not annotated_tasks:
        raise ValueError(
            "Aucune page annotée n'a été trouvée dans l'export Label Studio."
        )

    entity_labels = extract_entity_labels(annotated_tasks)
    label2id, id2label = build_label_vocab(entity_labels)

    print(f"\nLabels BIO : {len(label2id)}")
    print(label2id)

    examples = []

    for task in annotated_tasks:
        example = build_layoutlm_example_from_task(
            task=task,
            ocr_dir=OCR_DIR,
        )

        example["ner_tags"] = encode_labels(
            ner_tags=example["ner_tags"],
            label2id=label2id,
        )

        examples.append(example)

        print(
            f"✓ {example['id']} "
            f"({len(example['tokens'])} tokens)"
        )

    print(f"\n{len(examples)} exemple(s) généré(s)")

    dataset = Dataset.from_list(examples)

    bio_distribution = Counter()

    for example in examples:
        for tag_id in example["ner_tags"]:
            bio_distribution[id2label[tag_id]] += 1

    print("\n===== Distribution réelle des labels BIO =====")

    for label, count in sorted(bio_distribution.items()):
        print(f"{label:<25} : {count}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.save_to_disk(
        str(DATASET_DIR)
    )

    labels_payload = {
        "entity_labels": entity_labels,
        "label2id": label2id,
        "id2label": {
            str(label_id): label
            for label_id, label in id2label.items()
        },
    }

    with LABELS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            labels_payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Dataset sauvegardé dans : "
        f"{OUTPUT_DIR}"
    )
    print(
        f"Vocabulaire sauvegardé dans : "
        f"{LABELS_PATH}"
    )

    print(dataset)
    print(dataset.features)

    assert len(dataset) == len(examples)

    assert (
        dataset.features["ner_tags"]
        .feature
        .dtype
        == "int64"
    )

    assert all(
        0 <= tag_id < len(label2id)
        for tag_id in dataset[0]["ner_tags"]
    )


if __name__ == "__main__":
    main()