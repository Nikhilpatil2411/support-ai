# 1. Base Image
FROM python:3.10-slim

# 2. Working Directory
WORKDIR /app

# 3. System Dependencies (Standard tools install karo)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Requirements first
COPY requirements.txt .

# 5. Install Libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Code and Models (.pkl files)
COPY . .

# 7. Port Expose (3000 as you requested)
EXPOSE 3000

# 8. Run App
ENTRYPOINT ["streamlit", "run", "final_app.py", "--server.port=3000", "--server.address=0.0.0.0"]