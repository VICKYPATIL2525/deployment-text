# =============================================================================
# Mindspace — Text API (LightGBM, port 9000)
# =============================================================================
# Build context: deployment-text/
# docker build -t mindspace-text-api .
# docker run --env-file .env -p 9000:9000 mindspace-text-api
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API source and all required artifacts
COPY api_text_to_sentiment.py .
COPY pipeline_output/ pipeline_output/
COPY demo-api-input-data-sample/ demo-api-input-data-sample/

# API_KEY is injected at runtime via --env-file or -e, not baked into the image
EXPOSE 9000

CMD ["uvicorn", "api_text_to_sentiment:app", "--host", "0.0.0.0", "--port", "9000"]
