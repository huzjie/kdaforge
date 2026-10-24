# kdaforge: zero-dependency reference image
FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip

EXPOSE 8321

CMD ["python", "-m", "kdaforge", "serve"]
