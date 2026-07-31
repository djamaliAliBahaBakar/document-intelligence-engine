import json
import time
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pytesseract import Output

INPUT_DIR = Path("benchmarks/ocr/input")
OUTPUT_DIR = Path("benchmarks/ocr/outputs/tesseract")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for pdf_path in sorted(INPUT_DIR.glob("*.pdf")):

    print("=" * 80)
    print(pdf_path.name)

    start = time.perf_counter()

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        # Conversion PDF -> Image (200 dpi environ)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples,
        )

        data = pytesseract.image_to_data(
            image,
            lang="fra",
            output_type=Output.DICT,
            config="--oem 1 --psm 3",
        )

        words = []

        for i in range(len(data["text"])):

            text = data["text"][i].strip()

            if text == "":
                continue

            confidence = float(data["conf"][i])

            if confidence < 0:
                continue

            left = int(data["left"][i])
            top = int(data["top"][i])
            width = int(data["width"][i])
            height = int(data["height"][i])

            words.append(
                {
                    "text": text,
                    "confidence": confidence / 100,
                    "bbox": [
                        left,
                        top,
                        left + width,
                        top + height,
                    ],
                }
            )

        pages.append(
            {
                "page_number": page_number,
                "image_width": image.width,
                "image_height": image.height,
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
        "engine": "tesseract",
        "document": pdf_path.name,
        "duration_seconds": round(duration, 3),
        "page_count": len(pages),
        "pages": pages,
    }

    output_path = OUTPUT_DIR / f"{pdf_path.stem}_tesseract.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Temps : {duration:.2f} secondes")
    print(f"Résultat sauvegardé : {output_path}")