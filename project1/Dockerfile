FROM python:3.10-slim
COPY train.py reqs.txt ./
RUN pip install -r reqs.txt
ENTRYPOINT ["python", "train.py"]