"""
Appelle l'API Mistral OCR sur tous les PDF d'un dossier, et sauvegarde
un JSON par devis dans ./data/ocr_mistralocr/, au même format que
Tesseract/PaddleOCR (compatible avec le notebook benchmark_ocr_devis.ipynb).

Prérequis :
    pip install mistralai
    export MISTRAL_API_KEY="ta_clé"   (récupérée sur console.mistral.ai)

Structure attendue :
    ./data/devis_pdfs/devis_1.pdf
    ./data/devis_pdfs/devis_2.pdf
    ...
    -> génère ./data/ocr_mistralocr/devis_1.json, devis_2.json, ...

Usage :
    python3 appeler_mistral_ocr.py
"""

import os
import re
import json
import time
from pathlib import Path
from mistralai.client import Mistral



INPUT_DIR = Path("benchmarks/ocr/input")
OUTPUT_DIR = Path("benchmarks/ocr/outputs/mistralocr")
# On épingle la version exacte plutôt que "mistral-ocr-latest" : un alias "latest"
# peut changer de modèle sous-jacent sans prévenir, ce qui casserait la
# reproductibilité d'un benchmark. "mistral-ocr-latest" pointe aujourd'hui vers
# OCR 4 (mistral-ocr-4-0), mais autant figer la version utilisée explicitement.
MODEL = "mistral-ocr-4-0"


def call_mistral_ocr(client, pdf_path, model=MODEL):
    with open(pdf_path, "rb") as f:
        uploaded = client.files.upload(
            file={"file_name": os.path.basename(pdf_path), "content": f},
            purpose="ocr",
        )
    signed_url = client.files.get_signed_url(file_id=uploaded.id)

    response = client.ocr.process(
        model=model,
        document={"type": "document_url", "document_url": signed_url.url},
        include_blocks=True,                      # bounding boxes + labels de bloc (titre, tableau, texte...)
        confidence_scores_granularity="word",     # valeurs valides : "word" ou "page" (pas "block")
    )
    return response


def clean_markdown(text):
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"[#*_`]", "", text)
    text = re.sub(r"-{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def convert_to_common_format(response, duration):
    """
    Structure réelle confirmée :
    - page.confidence_scores = {
          "average_page_confidence_score": ..,
          "minimum_page_confidence_score": ..,
          "word_confidence_scores": [{"text": .., "confidence": .., "start_index": ..}, ...]
      }
      -> la source la plus fine (mot par mot), comparable à Tesseract, mais sans bbox.

    - page.blocks = [{"top_left_x":.., "top_left_y":.., "bottom_right_x":.., "bottom_right_y":..,
                      "content": "texte", "type": "header"/"text"/"table"/...}, ...]
      -> granularité paragraphe, avec bbox + type, mais sans confidence.

    On utilise word_confidence_scores comme liste principale "words" (texte + confiance,
    compatible avec le format des autres moteurs), et on garde les blocks à part sous une
    clé "blocks" pour ne rien perdre (utile plus tard si besoin de bbox/type).
    """
    pages_out = []
    for page in response.pages:
        page_dict = page.model_dump() if hasattr(page, "model_dump") else vars(page)

        conf_data = page_dict.get("confidence_scores") or {}
        word_scores = conf_data.get("word_confidence_scores") or []

        words = []
        if word_scores:
            for w in word_scores:
                words.append({
                    "text": clean_markdown(w.get("text") or ""),
                    "confidence": w.get("confidence"),
                    "bbox": None,  # pas de position associée à ce niveau de granularité
                })
        else:
            # Fallback : si confidence_scores absent, on retombe sur les blocks (moins fin)
            for block in (page_dict.get("blocks") or []):
                x0, y0 = block.get("top_left_x"), block.get("top_left_y")
                x1, y1 = block.get("bottom_right_x"), block.get("bottom_right_y")
                bbox_out = [x0, y0, x1 - x0, y1 - y0] if None not in (x0, y0, x1, y1) else None
                words.append({
                    "text": clean_markdown(block.get("content") or ""),
                    "confidence": None,
                    "bbox": bbox_out,
                })

        # Blocks conservés à part (bbox + type), en plus des "words"
        blocks_out = []
        for block in (page_dict.get("blocks") or []):
            x0, y0 = block.get("top_left_x"), block.get("top_left_y")
            x1, y1 = block.get("bottom_right_x"), block.get("bottom_right_y")
            bbox_out = [x0, y0, x1 - x0, y1 - y0] if None not in (x0, y0, x1, y1) else None
            blocks_out.append({
                "text": clean_markdown(block.get("content") or ""),
                "type": block.get("type"),
                "bbox": bbox_out,
            })

        pages_out.append({
            "page_number": page_dict.get("index", 0) + 1,
            "word_count": len(words),
            "words": words,
            "blocks": blocks_out,
            "avg_confidence": conf_data.get("average_page_confidence_score"),
            "min_confidence": conf_data.get("minimum_page_confidence_score"),
        })

    return {
        "engine": "mistralocr",
        "model": MODEL,
        "duration_seconds": round(duration, 2),
        "page_count": len(pages_out),
        "pages": pages_out,
    }


def main():
    if "MISTRAL_API_KEY" not in os.environ:
        raise SystemExit(
            "Variable d'environnement MISTRAL_API_KEY manquante.\n"
            "Lance d'abord : export MISTRAL_API_KEY='ta_clé'"
        )

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"Aucun PDF trouvé dans {input_dir}. Places-y tes devis d'abord.")
        return

    print(f"{len(pdf_paths)} PDF trouvés dans {input_dir}\n")

    for pdf_path in pdf_paths:
        devis_id = pdf_path.stem  # ex: devis_2.pdf -> devis_2
        output_path = output_dir / f"{devis_id}.json"

        if output_path.exists():
            print(f"[{devis_id}] déjà traité, ignoré (supprime le JSON pour relancer).")
            continue

        print(f"[{devis_id}] appel à l'API Mistral OCR...")
        start = time.time()
        try:
            response = call_mistral_ocr(client, pdf_path)
        except Exception as e:
            print(f"[{devis_id}] ERREUR : {e}")
            continue
        duration = time.time() - start

        result = convert_to_common_format(response, duration)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[{devis_id}] OK en {duration:.1f}s -> {output_path}")

    print("\nTerminé. Les JSON sont dans", output_dir)


if __name__ == "__main__":
    main()