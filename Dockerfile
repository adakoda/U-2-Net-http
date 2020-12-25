FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime
RUN apt-get update && apt-get install -y \
	python3-pip software-properties-common wget && \
	rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY U-2-Net ./U-2-Net
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY *.py ./
ENV PORT 80
CMD exec gunicorn --bind :$PORT --workers 1 main:app
