FROM python:3.10

WORKDIR /report1

COPY requirements.txt requirements.txt
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false"]