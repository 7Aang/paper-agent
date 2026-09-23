# Deployment

## Local FastAPI

```powershell
python -m pip install -r requirements.txt
uvicorn paper_agent.api:app --host 127.0.0.1 --port 8000
```

The API exposes `/health`, `/papers`, `/search`, and `/research`. LLM mode reads credentials only from environment variables.

## Docker

```powershell
docker compose up --build
```

`data/` and `runs/` are mounted as local volumes and remain outside the image. Keep `.env` local. The service runs as an unprivileged user.

## Limits

This project has been tested locally. No Kubernetes, managed database, multi-tenant authentication, or production load-test result is claimed.
The FastAPI routes are covered by local integration tests. Docker was unavailable in the current test environment, so image build status is `NOT RUN`.
