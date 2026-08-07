FROM python:3.13-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-fra \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt .

RUN pip install --no-cache-dir \
    -r requirements-runtime.txt

COPY app ./app

COPY models/layoutlmv3-photovoltaic-full-split-70-15-15/final \
     ./models/layoutlmv3-photovoltaic-full-split-70-15-15/final

EXPOSE 8000

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]