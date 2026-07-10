import json
import time
from pathlib import Path

import fitz  # PyMuPDF
import easyocr
from PIL import Image

INPUT_DIR = Path("benchmarks/ocr/input")
OUTPUT_DIR = Path("benchmarks/ocr/outputs/easyocr")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np

# Chargement du modèle (une seule fois)
reader = easyocr.Reader(
    ["fr"],
    gpu=False,
)

for pdf_path in sorted(INPUT_DIR.glob("*.pdf")):

    print("=" * 80)
    print(pdf_path.name)

    start = time.perf_counter()

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        # PDF -> image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples,
        )

        image_array = np.array(image)

        results = reader.readtext(image_array)

        words = []

        for bbox, text, confidence in results:

            x1 = int(bbox[0][0])
            y1 = int(bbox[0][1])

            x2 = int(bbox[2][0])
            y2 = int(bbox[2][1])

            words.append(
                {
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
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
            f"Page {page_number} : {len(words)} textes reconnus"
        )

    document.close()

    duration = time.perf_counter() - start

    output = {
        "engine": "easyocr",
        "document": pdf_path.name,
        "duration_seconds": round(duration, 3),
        "page_count": len(pages),
        "pages": pages,
    }

    output_path = OUTPUT_DIR / f"{pdf_path.stem}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Temps : {duration:.2f} secondes")
    print(f"Résultat sauvegardé : {output_path}")