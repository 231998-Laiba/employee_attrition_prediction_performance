# Production-style image for the Flask prediction API
FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY docs/model_report.json docs/model_report.json
EXPOSE 5000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/health')"
CMD ["gunicorn", "--chdir", "backend", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
