from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File
from docling.document_converter import DocumentConverter

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

from document_intelligence.parsing.markdown_tables import (
    extract_markdown_tables,
    parse_markdown_table,
    normalize_supplier_df_to_beluo,
    normalize_quantity,
    normalize_prices,
    to_beluo_json,
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

    tables = extract_markdown_tables(markdown)

    if not tables:
        return {
            "status": "needs_review",
            "reason": "No markdown table detected",
            "items": [],
        }

    table = tables[0]

    df = parse_markdown_table(table)
    df = normalize_supplier_df_to_beluo(df)
    df = normalize_quantity(df)
    df = normalize_prices(df)

    items = to_beluo_json(df)

    return {
        "status": "extracted",
        "filename": file.filename,
        "items_count": len(items),
        "items": items,
    }