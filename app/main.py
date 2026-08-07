import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from app.model import build_json, model, predict_page
from app.model import predict_pdf


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



@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
) -> PredictionResponse:

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=415,
            detail="Seuls les fichiers PDF sont acceptés.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Le fichier PDF est vide.",
        )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_path = temporary_file.name

        result = predict_pdf(
            temporary_path
        )

        return PredictionResponse(
            **result
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Échec de l'analyse du document : {error}",
        ) from error

    finally:
        if temporary_path is not None:
            Path(
                temporary_path
            ).unlink(
                missing_ok=True
            )