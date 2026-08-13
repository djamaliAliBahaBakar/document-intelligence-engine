import json
from pathlib import Path

from PIL import Image


OCR_DIR = Path("dataset/ocr")
IMAGES_DIR = Path("dataset/images")


def validate_document(ocr_path: Path) -> tuple[int, list[str]]:
    with ocr_path.open(encoding="utf-8") as file:
        ocr_data = json.load(file)

    document_name = ocr_path.stem.removesuffix("_tesseract")

    validated_pages = 0
    errors: list[str] = []

    for page in ocr_data["pages"]:
        page_number = page["page_number"]

        image_path = (
            IMAGES_DIR
            / f"{document_name}_page_{page_number}.png"
        )

        if not image_path.exists():
            errors.append(
                f"{document_name}, page {page_number}: "
                f"image absente"
            )
            continue

        expected_size = (
            page["image_width"],
            page["image_height"],
        )

        with Image.open(image_path) as image:
            actual_size = image.size

        if actual_size != expected_size:
            errors.append(
                f"{document_name}, page {page_number}: "
                f"image={actual_size}, OCR={expected_size}"
            )
            continue

        validated_pages += 1

    return validated_pages, errors


def main() -> None:
    ocr_paths = sorted(OCR_DIR.glob("*_tesseract.json"))

    if not ocr_paths:
        raise FileNotFoundError(
            f"Aucun fichier OCR trouvé dans {OCR_DIR.resolve()}"
        )

    total_validated = 0
    all_errors: list[str] = []

    for ocr_path in ocr_paths:
        validated_pages, errors = validate_document(ocr_path)

        total_validated += validated_pages
        all_errors.extend(errors)

        print(
            f"{ocr_path.name}: "
            f"{validated_pages} page(s) validée(s)"
        )

    if all_errors:
        print("\nErreurs détectées :")

        for error in all_errors:
            print(f"  - {error}")

        raise SystemExit(
            f"\nValidation échouée : "
            f"{len(all_errors)} erreur(s)."
        )

    print(
        f"\nValidation réussie : "
        f"{total_validated} image(s) cohérente(s)."
    )


if __name__ == "__main__":
    main()