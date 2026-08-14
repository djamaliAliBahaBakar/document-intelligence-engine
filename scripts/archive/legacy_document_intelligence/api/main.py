from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File
from docling.document_converter import DocumentConverter

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

from document_intelligence.parsing.markdown_tables import (
    find_quote_table,
    parse_markdown_table,
    normalize_supplier_df_to_beluo,
    normalize_quantity,
    normalize_prices,
    to_beluo_json,
    keep_business_rows
)

app = FastAPI(title="Beluo Document Intelligence API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract_quote(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {
            "status": "error",
            "message": "Only PDF files are supported for now",
        }

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(str(tmp_path))

    markdown = result.document.export_to_markdown()

    with open("devis3.md", "w", encoding="utf-8") as f:
        f.write(markdown)

    print("Markdown sauvegardé")

    try:
        table = find_quote_table(markdown)
    except ValueError:
        return {
            "status": "needs_review",
            "reason": "No quote table detected",
            "items": [],
        }
    
    # <<< AJOUTE ICI >>>
    print("=" * 80)
    print(table)
    print("=" * 80)


    df = parse_markdown_table(table)
    df = normalize_supplier_df_to_beluo(df)
    df = normalize_quantity(df)
    df = normalize_prices(df)
    df = keep_business_rows(df)


    items = to_beluo_json(df)

    return {
        "status": "extracted",
        "filename": file.filename,
        "items_count": len(items),
        "items": items,
    }