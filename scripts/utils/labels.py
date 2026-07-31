from collections.abc import Iterable


def extract_entity_labels(tasks: list[dict]) -> list[str]:
    labels: set[str] = set()

    for task in tasks:
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue

                rectangle_labels = result.get("value", {}).get(
                    "rectanglelabels",
                    [],
                )

                labels.update(rectangle_labels)

    return sorted(labels)


def build_label_vocab(
    entity_labels: Iterable[str],
) -> tuple[dict[str, int], dict[int, str]]:
    bio_labels = ["O"]

    for entity_label in sorted(set(entity_labels)):
        bio_labels.append(f"B-{entity_label}")
        bio_labels.append(f"I-{entity_label}")

    label2id = {
        label: index
        for index, label in enumerate(bio_labels)
    }

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    return label2id, id2label


def encode_labels(
    ner_tags: list[str],
    label2id: dict[str, int],
) -> list[int]:
    unknown_labels = sorted(
        set(ner_tags) - set(label2id)
    )

    if unknown_labels:
        raise ValueError(
            f"Labels BIO inconnus : {unknown_labels}"
        )

    return [
        label2id[label]
        for label in ner_tags
    ]