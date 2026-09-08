FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# RQ worker imports src.tasks.run_one_segment from /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
