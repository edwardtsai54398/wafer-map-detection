FROM python:3.13-slim

WORKDIR /app

RUN pip install uv
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --group backend

COPY . /app

EXPOSE 8080

CMD [".venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
