from pathlib import Path
from uuid import uuid4
import json
from collections import Counter

def bbox_to_percentages(
    bbox: list[int],
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    x1, y1, x2, y2 = bbox

    return {
        "x": 100 * x1 / image_width,
        "y": 100 * y1 / image_height,
        "width": 100 * (x2 - x1) / image_width,
        "height": 100 * (y2 - y1) / image_height,
    }







def build_regions(
    words: list[dict],
    image_width: int,
    image_height: int,
) -> list[dict]:
    regions: list[dict] = []

    for word in words:
        text = word["text"].strip()

        if not text:
            continue

        bbox = word["bbox"]

        coordinates = bbox_to_percentages(
            bbox=bbox,
            image_width=image_width,
            image_height=image_height,
        )

        region_id = uuid4().hex



        transcription_region = {
            "id": region_id,
            "from_name": "transcription",
            "to_name": "image",
            "type": "textarea",
            "value": {
                "text": [text],
            },
            "original_width": image_width,
            "original_height": image_height,
            "image_rotation": 0,
        }

        regions.append(
                transcription_region,
        )

    return regions


def build_task(
    image_path: str,
    regions: list[dict],
    model_version: str = "tesseract-v1",
) -> dict:
    image_url = f"http://localhost:8001/{image_path}"
    task = {
    "data": {
        "image": image_url,
    },
    "predictions": [
        {
            "model_version": model_version,
            "score": 1.0,
            "result": regions,
        }
    ],
}
    return task


from pathlib import Path
import json
from urllib.parse import urlparse, unquote


def load_export(export_path: Path) -> list[dict]:
    with export_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("L'export Label Studio doit contenir une liste de tâches.")

    return data


def extract_image_name(task: dict) -> str:
    image_url = task.get("data", {}).get("image")

    if not image_url:
        raise ValueError("La tâche Label Studio ne contient pas data.image.")

    parsed_url = urlparse(image_url)
    return Path(unquote(parsed_url.path)).name


import re
from pathlib import Path


def extract_document_id(image_name: str) -> str:
    stem = Path(image_name).stem

    match = re.search(r"(devis_\d+)_page_\d+$", stem)

    if not match:
        raise ValueError(
            f"Impossible d'extraire l'identifiant du document depuis : {image_name}"
        )

    return match.group(1)





def get_ocr_json_path(document_id: str, ocr_dir: Path) -> Path:
    path = ocr_dir / f"{document_id}_tesseract.json"

    if not path.exists():
        raise FileNotFoundError(f"Fichier OCR introuvable : {path}")

    return path





def load_ocr_page(ocr_path: Path, page_number: int) -> dict:
    with ocr_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    pages = data.get("pages", [])

    for page in pages:
        if page.get("page_number") == page_number:
            return page

    raise ValueError(
        f"Page {page_number} introuvable dans le fichier OCR : {ocr_path}"
    )


import re
from pathlib import Path


def extract_page_number(image_name: str) -> int:
    stem = Path(image_name).stem

    match = re.search(r"_page_(\d+)$", stem)

    if not match:
        raise ValueError(
            f"Impossible d'extraire le numéro de page depuis : {image_name}"
        )

    return int(match.group(1))


def extract_annotations(task: dict) -> list[dict]:
    annotations = task.get("annotations", [])

    if not annotations:
        return []

    results = annotations[0].get("result", [])
    extracted = []

    for result in results:
        if result.get("type") != "rectanglelabels":
            continue

        value = result.get("value", {})
        labels = value.get("rectanglelabels", [])

        if not labels:
            continue

        extracted.append(
            {
                "label": labels[0],
                "x_percent": value["x"],
                "y_percent": value["y"],
                "width_percent": value["width"],
                "height_percent": value["height"],
                "original_width": result["original_width"],
                "original_height": result["original_height"],
            }
        )

    return extracted


def annotation_to_ocr_bbox(
    annotation: dict,
    image_width: int,
    image_height: int,
) -> list[int]:
    x1 = round(annotation["x_percent"] / 100 * image_width)
    y1 = round(annotation["y_percent"] / 100 * image_height)

    x2 = round(
        (annotation["x_percent"] + annotation["width_percent"])
        / 100
        * image_width
    )
    y2 = round(
        (annotation["y_percent"] + annotation["height_percent"])
        / 100
        * image_height
    )

    return [x1, y1, x2, y2]

def bbox_center(bbox: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def bbox_contains_point(
    bbox: list[int],
    point: tuple[float, float],
) -> bool:
    x1, y1, x2, y2 = bbox
    x, y = point

    return x1 <= x <= x2 and y1 <= y <= y2


def find_words_in_annotation(
    words: list[dict],
    annotation_bbox: list[int],
) -> list[dict]:
    matched_words = []

    for word in words:
        center = bbox_center(word["bbox"])

        if bbox_contains_point(annotation_bbox, center):
            matched_words.append(word)

    return matched_words


def assign_bio_labels(
    words: list[dict],
    annotations: list[dict],
    image_width: int,
    image_height: int,
) -> list[str]:
    labels = ["O"] * len(words)

    for annotation in annotations:
        annotation_bbox = annotation_to_ocr_bbox(
            annotation=annotation,
            image_width=image_width,
            image_height=image_height,
        )

        matched_indexes = []

        for index, word in enumerate(words):
            center = bbox_center(word["bbox"])

            if bbox_contains_point(annotation_bbox, center):
                matched_indexes.append(index)

        for position, word_index in enumerate(matched_indexes):
            prefix = "B" if position == 0 else "I"
            labels[word_index] = f"{prefix}-{annotation['label']}"

    return labels


def normalize_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
) -> list[int]:
    x1, y1, x2, y2 = bbox

    normalized = [
        round(1000 * x1 / image_width),
        round(1000 * y1 / image_height),
        round(1000 * x2 / image_width),
        round(1000 * y2 / image_height),
    ]

    return [
        max(0, min(1000, value))
        for value in normalized
    ]


def build_layoutlm_example(
    document_id: str,
    page_number: int,
    page: dict,
    annotations: list[dict],
) -> dict:
    words = page["words"]

    tokens = [word["text"] for word in words]

    bboxes = [
        normalize_bbox(
            bbox=word["bbox"],
            image_width=page["image_width"],
            image_height=page["image_height"],
        )
        for word in words
    ]

    ner_tags = assign_bio_labels(
        words=words,
        annotations=annotations,
        image_width=page["image_width"],
        image_height=page["image_height"],
    )

    if not (len(tokens) == len(bboxes) == len(ner_tags)):
        raise ValueError(
            "tokens, bboxes et ner_tags doivent avoir la même longueur"
        )

    image_path = f"dataset/images/{document_id}_page_{page_number}.png"

    return {
        "id": f"{document_id}_page_{page_number}",
        "tokens": tokens,
        "bboxes": bboxes,
        "ner_tags": ner_tags,
        "image_path": image_path
    }

def build_layoutlm_example_from_task(
    task: dict,
    ocr_dir: Path,
    
) -> dict:
    image_name = extract_image_name(task)
    document_id = extract_document_id(image_name)
    page_number = extract_page_number(image_name)

    ocr_path = get_ocr_json_path(
        document_id=document_id,
        ocr_dir=ocr_dir,
    )

    page = load_ocr_page(
        ocr_path=ocr_path,
        page_number=page_number,
    )

    annotations = extract_annotations(task)

    print(
        Counter(
            annotation["label"]
            for annotation in annotations
        )
    )

    return build_layoutlm_example(
        document_id=document_id,
        page_number=page_number,
        page=page,
        annotations=annotations,
    )