FROM python:3.9-slim-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

# port 8501 is used by streamlit
EXPOSE 8501
CMD ["streamlit", "run", "src/main.py"]
