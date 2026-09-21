FROM python:3.12-slim

RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ ./src/
COPY config/ ./config/
# The checkpoint is intentionally gitignored. CI/deployment must provide it in
# the Docker build context or mount/download it to PIGNN_CHECKPOINT at startup.
COPY checkpoints/ ./checkpoints/
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    PORT=8080 \
    PIGNN_MINE_GEOMETRY_PATH=/app/config/mine_geometry.json \
    PIGNN_CHECKPOINT=/app/checkpoints/pignn-0.1.1.pt
EXPOSE 8080
CMD ["uvicorn", "pignn.service.main:app", "--host", "0.0.0.0", "--port", "8080"]
