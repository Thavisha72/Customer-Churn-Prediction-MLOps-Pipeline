# Use a slim Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional but safe for many ML libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code
COPY . /app

# Expose API port
EXPOSE 8000

# Start FastAPI (expects api/main.py with app = FastAPI())
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]