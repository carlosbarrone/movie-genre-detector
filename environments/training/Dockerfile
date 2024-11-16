FROM --platform=linux/amd64 python:3.9-bookworm
ARG ENV_NAME
WORKDIR /opt/ml/code/
COPY ./movie_detector /opt/ml/code/movie_detector
COPY ./mlops/steps/training.py /opt/ml/code/training.py
COPY ./environments/$ENV_NAME/requirements.txt /opt/ml/code/requirements.txt
RUN apt-get update
RUN apt-get install -y --no-install-recommends
RUN pip install -r requirements.txt --default-timeout=50000
ENV PYTHONPATH="/opt/ml/code:${PYTHONPATH}"
WORKDIR /opt/ml/code
ENV SAGEMAKER_PROGRAM training.py