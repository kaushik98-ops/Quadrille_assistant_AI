FROM python:3.12-slim

# Must match the Chroma server image tag. Pass it with --build-arg (Compose does this for you).
ARG CHROMA_VERSION
RUN test -n "$CHROMA_VERSION" || (echo "Set CHROMA_VERSION (see .env.example)" >&2 && exit 1)

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
# Pinning chromadb here keeps the client in step with the server container.
RUN pip install . "chromadb==${CHROMA_VERSION}" \
 && python -c "import assistant.api.main as m; assert (m.STATIC / 'index.html').is_file() and (m.STATIC / 'app.js').is_file(), 'chat UI files missing from the package'"

COPY scripts ./scripts
COPY docker/healthcheck.py /usr/local/bin/healthcheck.py

# These folders become the volume mount points. Creating them as the app user means a
# fresh named volume is initialised with the right owner.
RUN mkdir -p /app/data/pdfs /app/data/sandbox /home/app/.cache \
 && chown -R app:app /app/data /home/app/.cache

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD ["python", "/usr/local/bin/healthcheck.py"]

CMD ["python", "-m", "assistant.api"]
