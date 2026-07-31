from pathlib import Path
from scripts.utils.labelstudio import annotation_to_ocr_bbox
from scripts.utils.labelstudio import find_words_in_annotation
from scripts.utils.labelstudio import assign_bio_labels
from scripts.utils.labelstudio import build_layoutlm_example_from_task

from scripts.utils.labelstudio import (
    load_export,
    extract_image_name,
    extract_document_id,
    extract_page_number,
    get_ocr_json_path,
    load_ocr_page,
    extract_annotations,
    build_layoutlm_example,
)

EXPORT_PATH = Path(
    "notebooks/data/label_studio/label_studio_export.json"
)


def main():
    tasks = load_export(EXPORT_PATH)

    print(f"Nombre de tâches : {len(tasks)}")

    first = tasks[0]

    from pprint import pprint

    annotations = first.get("annotations", [])

    print(f"Nombre d'annotations : {len(annotations)}")

    if annotations:
        results = annotations[0].get("result", [])
        print(f"Nombre de régions annotées : {len(results)}")
        pprint(results[:2])

    image = extract_image_name(first)

    document_id = extract_document_id(image)

    print(f"Document ID : {document_id}")

    assert document_id == "devis_2"

    assert len(tasks) > 0
    assert image.endswith(".png")



    OCR_DIR = Path("dataset/ocr")

    ocr_path = get_ocr_json_path(document_id, OCR_DIR)

    print(f"OCR : {ocr_path}")

    assert ocr_path.exists()

    page_number = extract_page_number(image)

    page = load_ocr_page(
        ocr_path=ocr_path,
        page_number=page_number,
    )

    print(f"Page : {page_number}")
    print(f"Dimensions : {page['image_width']}x{page['image_height']}")
    print(f"Nombre de mots OCR : {len(page['words'])}")

    assert page["page_number"] == 1
    assert page["image_width"] > 0
    assert page["image_height"] > 0
    assert len(page["words"]) > 0

    annotations = extract_annotations(first)

    print(f"Annotations extraites : {len(annotations)}")
    print(annotations[0])

    assert len(annotations) == 5
    assert annotations[0]["label"] == "DATE_DEVIS"
    assert 0 <= annotations[0]["x_percent"] <= 100
    assert 0 <= annotations[0]["y_percent"] <= 100

    first_annotation = annotations[0]

    annotation_bbox = annotation_to_ocr_bbox(
        annotation=first_annotation,
        image_width=page["image_width"],
        image_height=page["image_height"],
    )

    print(f"BBox OCR annotation : {annotation_bbox}")

    x1, y1, x2, y2 = annotation_bbox

    assert 0 <= x1 < x2 <= page["image_width"]
    assert 0 <= y1 < y2 <= page["image_height"]

    matched_words = find_words_in_annotation(
    words=page["words"],
    annotation_bbox=annotation_bbox,
)

    matched_texts = [word["text"] for word in matched_words]

    print(f"Mots dans {first_annotation['label']} : {matched_texts}")

    assert len(matched_words) > 0

    for annotation in annotations:
        bbox = annotation_to_ocr_bbox(
            annotation=annotation,
            image_width=page["image_width"],
            image_height=page["image_height"],
        )

        matched_words = find_words_in_annotation(
            words=page["words"],
            annotation_bbox=bbox,
        )

        print(f"\n{annotation['label']}")
        print(f"Annotation bbox : {bbox}")

        for word in matched_words:
            print(
                f"  text={word['text']!r}, "
                f"confidence={word['confidence']}, "
                f"bbox={word['bbox']}"
            )

        assert matched_words, (
            f"Aucun mot OCR trouvé pour {annotation['label']}"
        )

    bio_labels = assign_bio_labels(
    words=page["words"],
    annotations=annotations,
    image_width=page["image_width"],
    image_height=page["image_height"],
    )

    assert len(bio_labels) == len(page["words"])

    for word, label in zip(page["words"], bio_labels):
        if label != "O":

            print(f"{word['text']!r} → {label}")


    example = build_layoutlm_example(
    document_id=document_id,
    page_number=page_number,
    page=page,
    annotations=annotations,
)

    print(f"Example ID : {example['id']}")
    print(f"Nombre de tokens : {len(example['tokens'])}")
    print(f"Nombre de bboxes : {len(example['bboxes'])}")
    print(f"Nombre de labels : {len(example['ner_tags'])}")

    assert example["id"] == "devis_2_page_1"
    assert len(example["tokens"]) == 173
    assert len(example["tokens"]) == len(example["bboxes"])
    assert len(example["tokens"]) == len(example["ner_tags"])

    for bbox in example["bboxes"]:
        assert len(bbox) == 4
        assert all(0 <= value <= 1000 for value in bbox)

    example_from_task = build_layoutlm_example_from_task(
    task=first,
    ocr_dir=Path("dataset/ocr"),
)

    assert example_from_task == example

    print(
        f"Pipeline complet validé : "
        f"{example_from_task['id']}"
    )

    print("OK")


if __name__ == "__main__":
    main()