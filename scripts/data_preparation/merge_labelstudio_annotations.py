import json
from pathlib import Path


INITIAL_PATH = Path(
    "label_studio_data/export/annotations.json"
)

NEW_PATH = Path(
    "label_studio_data/export/annotations_new.json"
)

OUTPUT_PATH = Path(
    "label_studio_data/export/annotations_all.json"
)


def load_json_list(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"{path} ne contient pas une liste JSON."
        )

    return data


def main() -> None:
    initial_annotations = load_json_list(INITIAL_PATH)
    new_annotations = load_json_list(NEW_PATH)

    merged_annotations = (
        initial_annotations
        + new_annotations
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            merged_annotations,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "Anciennes annotations :",
        len(initial_annotations),
    )
    print(
        "Nouvelles annotations :",
        len(new_annotations),
    )
    print(
        "Total fusionné :",
        len(merged_annotations),
    )
    print(
        "Fichier généré :",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()