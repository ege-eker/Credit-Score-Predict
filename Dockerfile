FROM python:3.12.3

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY models models
COPY *.py ./

CMD ["python","api.py"]