FROM --platform=linux/amd64 python:3.9-bookworm
ARG ENV_NAME
WORKDIR /opt/ml/code/
COPY ./movie_detector /opt/ml/code/movie_detector
COPY ./environments/inference/prerequisites /opt/ml/code
COPY ./mlops/steps/training.py /opt/ml/code/training.py
COPY ./mlops/steps/predict.py /opt/ml/code/predict.py
COPY ./environments/$ENV_NAME/requirements.txt /opt/ml/code/requirements.txt
RUN apt-get update
RUN apt-get install -y --no-install-recommends wget python3-pip python3-setuptools nginx ca-certificates
RUN pip install -r requirements.txt --default-timeout=50000
RUN rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN chmod +x /opt/ml/code/serve
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PYTHONPATH="/opt/ml/code:${PYTHONPATH}"
ENV PATH="/opt/ml/code:${PATH}"
WORKDIR /opt/ml/code
ENV SAGEMAKER_PROGRAM training.py