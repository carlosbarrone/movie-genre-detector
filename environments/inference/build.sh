#!/bin/bash
env_name=$(basename "$PWD")
algorithm_name="movie-genre-detector-$env_name"
cd ../../
account=$(aws sts get-caller-identity --query Account --output text)
region=${AWS_REGION}
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com
docker build --no-cache -t ${algorithm_name} --build-arg ENV_NAME=$env_name -f ./environments/inference/Dockerfile .
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}