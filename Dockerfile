FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py app.py api.py frontend.py train.py README.md ./
COPY data/ data/
COPY models/ models/

# Default: run the API. Platforms like Render/Railway/Cloud Run set $PORT.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:api --host 0.0.0.0 --port ${PORT}"]
