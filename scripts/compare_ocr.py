import csv
import json
from pathlib import Path


OUTPUT_ROOT = Path("benchmarks/ocr/outputs")
REPORT_PATH = Path("benchmarks/ocr/ocr_summary.csv")

ENGINES = ["paddle", "tesseract", "easyocr"]


def load_ocr_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_full_text(result: dict) -> str:
    texts = []

    for page in result.get("pages", []):
        for word in page.get("words", []):
            text = str(word.get("text", "")).strip()

            if text:
                texts.append(text)

    return " ".join(texts)


def get_average_confidence(result: dict) -> float | None:
    confidences = []

    for page in result.get("pages", []):
        for word in page.get("words", []):
            confidence = word.get("confidence")

            if confidence is not None:
                confidences.append(float(confidence))

    if not confidences:
        return None

    return sum(confidences) / len(confidences)


def get_word_count(result: dict) -> int:
    return sum(
        len(page.get("words", []))
        for page in result.get("pages", [])
    )


rows = []

for engine in ENGINES:
    engine_dir = OUTPUT_ROOT / engine

    if not engine_dir.exists():
        print(f"Dossier absent : {engine_dir}")
        continue

    for json_path in sorted(engine_dir.glob("*.json")):
        result = load_ocr_result(json_path)

        full_text = get_full_text(result)
        average_confidence = get_average_confidence(result)

        rows.append(
            {
                "document": result.get(
                    "document",
                    f"{json_path.stem}.pdf",
                ),
                "engine": engine,
                "duration_seconds": result.get(
                    "duration_seconds",
                ),
                "page_count": result.get("page_count"),
                "word_count": get_word_count(result),
                "average_confidence": (
                    round(average_confidence, 4)
                    if average_confidence is not None
                    else ""
                ),
                "text_preview": full_text[:300],
            }
        )

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with REPORT_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "document",
            "engine",
            "duration_seconds",
            "page_count",
            "word_count",
            "average_confidence",
            "text_preview",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)

print(f"Rapport généré : {REPORT_PATH}")
print(f"{len(rows)} résultats analysés.")