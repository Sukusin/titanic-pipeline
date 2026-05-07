FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY src ./src
COPY configs ./configs
COPY data/processed ./data/processed
COPY Makefile .

CMD ["uv", "run", "python", "-m", "src.classic.make_leaderboard"]