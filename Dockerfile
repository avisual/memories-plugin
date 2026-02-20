# avisual memories - Docker container
# Build: docker build -t avisual-memories .
# Run: docker run -v memories-data:/data -p 8080:8080 avisual-memories

FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY README.md ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Production image
FROM python:3.13-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV MEMORIES_DB_PATH="/data/memories.db"

# Create data directory
RUN mkdir -p /data

# Volume for persistent storage
VOLUME ["/data"]

# Expose MCP server port (if running HTTP mode)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m memories health || exit 1

# Default command: run MCP server
ENTRYPOINT ["python", "-m", "memories"]
CMD ["serve"]

# Labels
LABEL org.opencontainers.image.title="avisual memories"
LABEL org.opencontainers.image.description="Brain-like memory system for AI agents"
LABEL org.opencontainers.image.source="https://github.com/avisual/memories"
LABEL org.opencontainers.image.licenses="MIT"
