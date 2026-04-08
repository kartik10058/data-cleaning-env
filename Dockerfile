FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir flask openenv-core>=0.2.0 || pip install --no-cache-dir flask

COPY . .

ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4.1-mini"

EXPOSE 7860

CMD ["python", "-m", "server.app"]
