from pathlib import Path

import fitz
import pytesseract
from PIL import Image
from pytesseract import Output


def preprocess_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Convertit un PDF en pages exploitables par LayoutLMv3.

    Retourne pour chaque page :
    - image
    - tokens
    - bboxes normalisées dans l'espace 0..1000
    """

    document = fitz.open(str(pdf_path))

    pages: list[dict] = []

    try:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False,
            )

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

            tokens: list[str] = []
            bboxes: list[list[int]] = []

            for index in range(len(data["text"])):
                text = data["text"][index].strip()

                if not text:
                    continue

                confidence = float(
                    data["conf"][index]
                )

                if confidence < 0:
                    continue

                left = int(data["left"][index])
                top = int(data["top"][index])
                width = int(data["width"][index])
                height = int(data["height"][index])

                x1 = left
                y1 = top
                x2 = left + width
                y2 = top + height

                normalized_bbox = [
                    int(1000 * x1 / image.width),
                    int(1000 * y1 / image.height),
                    int(1000 * x2 / image.width),
                    int(1000 * y2 / image.height),
                ]

                normalized_bbox = [
                    max(0, min(1000, value))
                    for value in normalized_bbox
                ]

                tokens.append(text)
                bboxes.append(normalized_bbox)

            pages.append(
                {
                    "page_number": page_number,
                    "image": image.copy(),
                    "tokens": tokens,
                    "bboxes": bboxes,
                }
            )

    finally:
        document.close()

    return pages