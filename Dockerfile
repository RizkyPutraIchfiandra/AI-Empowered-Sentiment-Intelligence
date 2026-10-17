# Hugging Face Spaces (Docker SDK) serves the Flask API only.
# The frontend is hosted separately on Netlify.
FROM python:3.12-slim

# Spaces runs the container as uid 1000 and mounts nothing writable at /, so
# everything lives in a home directory this user owns.
RUN useradd -m -u 1000 app
USER app
ENV PATH="/home/app/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /home/app

COPY --chown=app backend/requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=app backend/ ./

# Train at build time rather than committing 6 MB of binaries. The datasets are
# fetched here too, so the running container needs no outbound network.
RUN python train.py

EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/health').read()"

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "60", "app:app"]
