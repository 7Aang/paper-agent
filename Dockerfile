FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt pyproject.toml README.md ./
COPY paper_agent ./paper_agent
RUN pip install --no-cache-dir .
RUN useradd --create-home appuser && mkdir -p /app/data /app/runs && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "paper_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
