import json
import os
import time
from pathlib import Path

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "false"

from paddleocr import PaddleOCR


INPUT_DIR = Path("benchmarks/ocr/input")
OUTPUT_DIR = Path("benchmarks/ocr/outputs/paddle")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ocr = PaddleOCR(
    lang="fr",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)



for pdf_path in sorted(INPUT_DIR.glob("*.pdf")):
    print("=" * 80)
    print(pdf_path.name)

    start = time.perf_counter()
    results = ocr.predict(str(pdf_path))
    duration = time.perf_counter() - start

    pages = []

    for page_number, page_result in enumerate(results, start=1):
        texts = page_result.get("rec_texts", [])
        scores = page_result.get("rec_scores", [])
        boxes = page_result.get("rec_boxes", [])

        words = []

        for text, score, box in zip(texts, scores, boxes):
            words.append(
                {
                    "text": text,
                    "confidence": float(score),
                    "bbox": box.tolist(),
                }
            )

        pages.append(
            {
                "page_number": page_number,
                "word_count": len(words),
                "words": words,
            }
        )

        print(
            f"Page {page_number} : "
            f"{len(words)} textes reconnus"
        )

    output = {
        "engine": "paddleocr",
        "document": pdf_path.name,
        "duration_seconds": round(duration, 3),
        "page_count": len(pages),
        "pages": pages,
    }

    output_path = OUTPUT_DIR / f"{pdf_path.stem}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(f"Temps total : {duration:.2f} secondes")
    print(f"Résultat sauvegardé : {output_path}")