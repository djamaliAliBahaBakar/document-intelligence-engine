from pathlib import Path
import json

import torch
from datasets import load_from_disk
from PIL import Image
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)


MODEL_DIR = Path("models/layoutlmv3-photovoltaic/final")
DATASET_DIR = Path("dataset/processed/layoutlm/dataset")
LABELS_PATH = Path("dataset/processed/layoutlm/labels.json")


def main() -> None:
    dataset = load_from_disk(str(DATASET_DIR))

    with LABELS_PATH.open("r", encoding="utf-8") as file:
        labels_payload = json.load(file)

    id2label = {
        int(label_id): label
        for label_id, label in labels_payload["id2label"].items()
    }

    processor = LayoutLMv3Processor.from_pretrained(
        str(MODEL_DIR),
        apply_ocr=False,
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        str(MODEL_DIR)
    )

    model.eval()

    example = dataset[0]

    image = Image.open(example["image_path"]).convert("RGB")

    encoding = processor(
        image,
        example["tokens"],
        boxes=example["bboxes"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoding)

    predicted_ids = outputs.logits.argmax(dim=-1)[0].tolist()

    from collections import Counter

    counter = Counter(predicted_ids)

    print("\n===== Répartition des labels prédits =====")

    for label_id, count in sorted(counter.items()):
        print(f"{id2label[label_id]:20} : {count}")

        word_ids = encoding.word_ids(batch_index=0)

    print("\n===== Prédictions =====")

    previous_word_id = None

    for token_position, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        if word_id == previous_word_id:
            continue

        token = example["tokens"][word_id]
        predicted_label_id = predicted_ids[token_position]
        predicted_label = id2label[predicted_label_id]

        print(f"{token:<30} -> {predicted_label}")

        previous_word_id = word_id


if __name__ == "__main__":
    main()