# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.8-slim

ENV PYTHONUNBUFFERED True

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

<<<<<<< HEAD:Dockerfile
COPY app /app

CMD ["python", "main.py"]
=======
# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
>>>>>>> b255a312bfae2aa48fcbd72e09b3221e65b2cb52:dockerfile
