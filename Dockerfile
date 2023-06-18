FROM python:3.9-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    ffmpeg \
    libsm6 \
    libxext6

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

# port 8501 is used by streamlit
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
CMD ["streamlit", "run", "src/main.py"]
