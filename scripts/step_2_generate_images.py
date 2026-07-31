from pathlib import Path

import pymupdf
from PIL import Image


INPUT_DIR = Path("dataset/raw")
OUTPUT_DIR = Path("dataset/images")

RENDER_MATRIX = pymupdf.Matrix(2, 2)


def generate_images_from_pdf(
    pdf_path: Path,
    output_dir: Path,
) -> list[Path]:
    generated_images: list[Path] = []

    document = pymupdf.open(pdf_path)

    try:
        for page_index, page in enumerate(document):
            page_number = page_index + 1

            pixmap = page.get_pixmap(
                matrix=RENDER_MATRIX,
                alpha=False,
            )

            image = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples,
            )

            output_path = (
                output_dir
                / f"{pdf_path.stem}_page_{page_number}.png"
            )

            image.save(output_path)
            generated_images.append(output_path)

    finally:
        document.close()

    return generated_images


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(INPUT_DIR.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"Aucun fichier PDF trouvé dans {INPUT_DIR.resolve()}"
        )

    total_images = 0

    for pdf_path in pdf_paths:
        generated_images = generate_images_from_pdf(
            pdf_path=pdf_path,
            output_dir=OUTPUT_DIR,
        )

        total_images += len(generated_images)

        print(
            f"{pdf_path.name}: "
            f"{len(generated_images)} image(s) générée(s)"
        )

        for image_path in generated_images:
            print(f"  - {image_path}")

    print(
        f"\nTerminé : {len(pdf_paths)} PDF, "
        f"{total_images} image(s)."
    )


if __name__ == "__main__":
    main()