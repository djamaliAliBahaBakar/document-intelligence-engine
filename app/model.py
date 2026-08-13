
from typing import Any
import os

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
)

from collections import defaultdict
from app.preprocess import preprocess_pdf




MODEL_SOURCE = os.getenv(
    "MODEL_SOURCE",
    "djamali/layoutlmv3-photovoltaic",
)

MAX_LENGTH = 512


print("Chargement du modèle...")

processor = AutoProcessor.from_pretrained(
    MODEL_SOURCE,
    apply_ocr=False,
)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    MODEL_SOURCE,
)

model.eval()

print("Modèle chargé.")




def predict_pdf(pdf_path: str) -> dict:
    pages = preprocess_pdf(pdf_path)

    all_predictions = []

    for page in pages:
        predictions = predict_page(
            image=page["image"],
            tokens=page["tokens"],
            bboxes=page["bboxes"],
        )

        all_predictions.extend(predictions)

    return build_json(all_predictions)


def predict_page(
    image: Image.Image,
    tokens: list[str],
    bboxes: list[list[int]],
) -> list[dict[str, Any]]:
    if len(tokens) != len(bboxes):
        raise ValueError(
            "Le nombre de tokens doit être égal au nombre de bounding boxes."
        )

    image = image.convert("RGB")

    encoding = processor(
        images=image,
        text=tokens,
        boxes=bboxes,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoding)

    predicted_ids = np.argmax(
        outputs.logits.detach().cpu().numpy(),
        axis=-1,
    )[0]

    word_ids = encoding.word_ids(batch_index=0)

    predictions: list[dict[str, Any]] = []
    seen_word_ids: set[int] = set()

    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_word_ids:
            continue

        seen_word_ids.add(word_id)

        if word_id >= len(tokens):
            continue

        label_id = int(predicted_ids[token_index])
        label = model.config.id2label[label_id]

        predictions.append(
            {
                "token": tokens[word_id],
                "bbox": bboxes[word_id],
                "label": label,
            }
        )

    return predictions





def build_json(predictions: list[dict]) -> dict:
    """
    Transforme les prédictions BIO en JSON métier.
    """

    entities = defaultdict(list)

    current_label = None

    for prediction in predictions:
        label = prediction["label"]
        token = prediction["token"]

        if label == "O":
            continue

        if label.startswith("B-"):
            current_label = label[2:]
            entities[current_label].append(token)

        elif label.startswith("I-"):
            entity = label[2:]

            if entity == current_label:
                entities[current_label].append(token)

    return {
        "client": " ".join(entities["CLIENT"]) or None,
        "fournisseur": " ".join(entities["EMETTEUR_DEVIS"]) or None,
        "numero_devis": " ".join(entities["NUMERO_DEVIS"]) or None,
        "date_devis": " ".join(entities["DATE_DEVIS"]) or None,
        "montant_total": " ".join(entities["MONTANT_TOTAL"]) or None,
    }