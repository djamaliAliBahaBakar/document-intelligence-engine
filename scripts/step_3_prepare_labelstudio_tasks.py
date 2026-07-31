from pathlib import Path
import json


from scripts.utils.labelstudio import build_regions, build_task


OCR_DIR = Path("dataset/ocr")
IMAGES_DIR = Path("dataset/images")
OUTPUT_FILE = Path("label_studio_data/tasks.json")


def main() -> None:
    tasks: list[dict] = []

    ocr_files = sorted(OCR_DIR.glob("*_tesseract.json"))

    for ocr_file in ocr_files:
        with ocr_file.open("r", encoding="utf-8") as file:
            ocr = json.load(file)

        document_name = ocr_file.stem.replace("_tesseract", "")

        for page in ocr["pages"]:
            page_number = page["page_number"]

            image_path = (
                IMAGES_DIR
                / f"{document_name}_page_{page_number}.png"
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image introuvable : {image_path}"
                )

            regions = build_regions(
                words=page["words"],
                image_width=page["image_width"],
                image_height=page["image_height"],
            )

            task = build_task(
                image_path=str(image_path),
                regions=regions,
            )

            tasks.append(task)

        print(
            f"{ocr_file.name}: "
            f"{len(ocr['pages'])} tâche(s) générée(s)"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            tasks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nGénération réussie : "
        f"{len(tasks)} tâche(s) enregistrée(s) "
        f"dans {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()