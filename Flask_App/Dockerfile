FROM python:3.11-slim-buster
WORKDIR /app
COPY requirements.txt /app
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt
COPY . /app
EXPOSE 5000
CMD ["python", "app.py"]
