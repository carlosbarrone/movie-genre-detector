FROM --platform=linux/amd64 python:3.9-bookworm
ARG ENV_NAME
WORKDIR /opt/ml/processing/
COPY ./movie_detector /opt/ml/processing/movie_detector
COPY ./environments/$ENV_NAME/requirements.txt /opt/ml/processing/requirements.txt
RUN apt-get update
RUN apt-get install -y --no-install-recommends
RUN pip install -r requirements.txt --default-timeout=50000
ENV PYTHONPATH="/opt/ml/processing:${PYTHONPATH}"
WORKDIR /opt/ml/processing/input/