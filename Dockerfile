FROM python:3.12-slim

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev

COPY src ./src
COPY configs ./configs
COPY data ./data
COPY Makefile .

CMD ["python", "-m", "src.main", "--competition", "titanic", "--models", "logreg_l2", "--dataset-types", "original"]
