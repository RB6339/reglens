FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[api,llm]"

# pre-built chunks so the container starts without downloading the PDFs
COPY data/chunks.jsonl ./data/chunks.jsonl
COPY configs ./configs

EXPOSE 8000
CMD ["uvicorn", "reglens.api:app", "--host", "0.0.0.0", "--port", "8000"]
