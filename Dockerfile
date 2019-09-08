FROM python

WORKDIR /app

RUN pwd

COPY . .

RUN ["pip", "install", "--no-cache-dir", "-r", "requirements.txt"]
RUN ["python", "model.py"]