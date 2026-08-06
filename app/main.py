import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from app.model import build_json, model, predict_page


app = FastAPI(
    title="Document Intelligence API",
    version="0.1.0",
)

class PredictionResponse(BaseModel):
    client: str | None = None
    fournisseur: str | None = None
    numero_devis: str | None = None
    date_devis: str | None = None
    montant_total: str | None = None


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "document-intelligence-api",
        "model": "layoutlmv3",
        "model_loaded": model is not None,
    }

@app.post("/predict",response_model=PredictionResponse,
)
async def predict(
    image: UploadFile = File(...),
    ocr: UploadFile = File(...),
) -> dict[str, str | None]:
    allowed_image_types = {
        "image/png",
        "image/jpeg",
    }

    if image.content_type not in allowed_image_types:
        raise HTTPException(
            status_code=415,
            detail="L'image doit être au format PNG ou JPEG.",
        )

    try:
        ocr_content = await ocr.read()
        ocr_payload = json.loads(
            ocr_content.decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=400,
            detail="Le fichier OCR n'est pas un JSON valide.",
        ) from error

    tokens = ocr_payload.get("tokens")
    bboxes = ocr_payload.get("bboxes")

    if not isinstance(tokens, list) or not isinstance(
        bboxes,
        list,
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Le JSON OCR doit contenir les listes "
                "'tokens' et 'bboxes'."
            ),
        )

    if len(tokens) != len(bboxes):
        raise HTTPException(
            status_code=422,
            detail=(
                "Le nombre de tokens doit être égal "
                "au nombre de bounding boxes."
            ),
        )

    image_suffix = Path(
        image.filename or "document.png"
    ).suffix

    image_content = await image.read()

    if not image_content:
        raise HTTPException(
            status_code=400,
            detail="Le fichier image est vide.",
        )

    temporary_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=image_suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(image_content)
            temporary_path = temporary_file.name

        predictions = predict_page(
            image_path=temporary_path,
            tokens=tokens,
            bboxes=bboxes,
        )

        return build_json(predictions)

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Échec de l'inférence.",
        ) from error

    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(
                missing_ok=True
            )