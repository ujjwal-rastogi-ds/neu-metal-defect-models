FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY results/ ./results/

# Expose application port (Railway will set PORT at runtime)
EXPOSE 7860

# Set environment variables
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Default PORT for local runs; Railway overrides this
ENV PORT=7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

# Run the application; bind to provided PORT
CMD ["streamlit", "run", "app.py", "--server.port=${PORT}", "--server.address=0.0.0.0"]