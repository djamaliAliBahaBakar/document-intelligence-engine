from pathlib import Path
import re


PDF_DIR = Path("dataset/raw/new_raw")


for pdf_path in PDF_DIR.glob("*.pdf"):
    match = re.search(r"(\d+)", pdf_path.stem)

    if match is None:
        print(f"Ignoré : aucun numéro trouvé dans {pdf_path.name}")
        continue

    document_number = match.group(1)

    new_name = f"devis_new_{document_number}.pdf"
    new_path = pdf_path.with_name(new_name)

    if new_path.exists():
        print(f"Ignoré : {new_name} existe déjà")
        continue

    pdf_path.rename(new_path)

    print(f"{pdf_path.name} -> {new_name}")