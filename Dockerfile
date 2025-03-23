# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Install system dependencies (build tools, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app

EXPOSE 8501 8502

# By default, run the Streamlit app (this command can be overridden in docker-compose)
CMD ["streamlit", "run", "chatbot_streamlit.py"]
