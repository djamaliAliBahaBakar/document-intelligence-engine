from pathlib import Path

from datasets import load_from_disk
from PIL import Image
import torch

from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)


MODEL_DIR = Path("models/layoutlmv3-photovoltaic/final")
DATASET_DIR = Path("dataset/processed/layoutlm/dataset")

TARGET_PAGE_ID = "devis_10_page_1"


def find_example_by_id(dataset, page_id: str) -> dict:
    for example in dataset:
        if example["id"] == page_id:
            return example

    raise ValueError(f"Page introuvable : {page_id}")


def main() -> None:
    processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_DIR
    )

    dataset = load_from_disk(str(DATASET_DIR))

    example = find_example_by_id(
        dataset,
        TARGET_PAGE_ID,
    )

    image = Image.open(
        example["image_path"]
    ).convert("RGB")

    encoding = processor(
        image,
        example["tokens"],
        boxes=example["bboxes"],
        word_labels=example["ner_tags"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    model.eval()

    with torch.no_grad():
        outputs = model(**encoding)

    predicted_ids = outputs.logits.argmax(dim=-1)[0].tolist()
    true_ids = encoding["labels"][0].tolist()

    id2label = model.config.id2label

    print(f"\n===== Évaluation : {TARGET_PAGE_ID} =====")
    print(f"{'TOKEN':30} {'RÉEL':22} {'PRÉDIT':22}")
    print("-" * 78)

    word_ids = encoding.word_ids(batch_index=0)

    previous_word_id = None

    for token_index, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        if word_id == previous_word_id:
            continue

        previous_word_id = word_id

        true_id = true_ids[token_index]
        predicted_id = predicted_ids[token_index]

        if true_id == -100:
            continue

        token = example["tokens"][word_id]
        true_label = id2label[true_id]
        predicted_label = id2label[predicted_id]

        if true_label != "O" or predicted_label != "O":
            print(
                f"{token[:29]:30} "
                f"{true_label:22} "
                f"{predicted_label:22}"
            )


if __name__ == "__main__":
    main()