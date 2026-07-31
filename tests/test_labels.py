from scripts.utils.labels import (
    build_label_vocab,
    encode_labels,
    extract_entity_labels,
)
from scripts.utils.labelstudio import load_export

from pathlib import Path


EXPORT_PATH = Path(
    "notebooks/data/label_studio/label_studio_export.json"
)


def main() -> None:
    tasks = load_export(EXPORT_PATH)

    entity_labels = extract_entity_labels(tasks)

    print("Labels métier :", entity_labels)

    label2id, id2label = build_label_vocab(entity_labels)

    print("label2id :", label2id)
    print("id2label :", id2label)

    assert entity_labels == [
        "DATE_DEVIS",
        "FOURNISSEUR",
        "MONTANT_TOTAL",
        "NUMERO_DEVIS",
        "SOCIETE",
    ]

    assert label2id["O"] == 0
    assert id2label[0] == "O"

    encoded = encode_labels(
        [
            "O",
            "B-DATE_DEVIS",
            "I-DATE_DEVIS",
            "B-SOCIETE",
        ],
        label2id,
    )

    print("Labels encodés :", encoded)

    assert encoded == [
        label2id["O"],
        label2id["B-DATE_DEVIS"],
        label2id["I-DATE_DEVIS"],
        label2id["B-SOCIETE"],
    ]

    print("OK")


if __name__ == "__main__":
    main()