import os
import boto3
from datetime import datetime

from sagemaker import Session
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.model import Model
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv('../.env')
    pipeline_name = 'MovieGenreDetector'
    sagemaker_session = Session()
    sagemaker_role = os.environ['SAGEMAKER_ROLE']
    pipeline_session = PipelineSession()
    s3_project_bucket = os.environ['S3_BUCKET']
    region = sagemaker_session.boto_region_name
    account_id = boto3.client('sts').get_caller_identity()['Account']
    ecr_client = boto3.client('ecr', region_name=region)
    repositories = ecr_client.describe_repositories()
    uri_base = f'{account_id}.dkr.ecr.{region}.amazonaws.com/movie-genre-detector'

    learning_rate_param = ParameterFloat(name="learning_rate", default_value=0.001)
    batch_size_param = ParameterInteger(name="batch_size", default_value=64)
    epochs_param = ParameterInteger(name="epochs", default_value=100)
    weight_decay_param = ParameterFloat(name="weight_decay", default_value=1e-6)
    threshold_param = ParameterFloat(name="threshold", default_value=0.4)
    hidden_layer_1_param = ParameterInteger(name="hidden_layer_1", default_value=256)
    hidden_layer_2_param = ParameterInteger(name="hidden_layer_2", default_value=96)

    custom_processor = ScriptProcessor(
        role=sagemaker_role,
        image_uri=f'{uri_base}-processing',
        instance_type='ml.t3.large',
        instance_count=1,
        base_job_name='custom-processor',
        max_runtime_in_seconds=900,
        command=['python3']
    )

    custom_estimator = Estimator(
        role=sagemaker_role,
        sagemaker_session=sagemaker_session,
        image_uri=f'{uri_base}-training',
        instance_type='ml.t3.large',
        instance_count=1,
        hyperparameters={
            'learning_rate': learning_rate_param,
            'batch_size': batch_size_param,
            'epochs': epochs_param,
            'weight_decay': weight_decay_param,
            'threshold': threshold_param,
            'hidden_layer_1': hidden_layer_1_param,
            'hidden_layer_2': hidden_layer_2_param
        }
    )

    step_process = ProcessingStep(
        name='processing',
        processor=custom_processor,
        inputs=[
            ProcessingInput(source=f's3://{s3_project_bucket}/data/raw/', destination="/opt/ml/processing/input")
        ],
        outputs=[
            ProcessingOutput(output_name='train', source='/opt/ml/processing/output/train/',
                             destination=f's3://{s3_project_bucket}/data/processed/train/'),
            ProcessingOutput(output_name='validation', source='/opt/ml/processing/output/validation/',
                             destination=f's3://{s3_project_bucket}/data/processed/validation/')
        ],
        code='./steps/processing.py'
    )

    step_train = TrainingStep(
        name='training',
        estimator=custom_estimator,
        inputs={
            'train': TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri,
                content_type="text/csv"),
            'validation': TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs['validation'].S3Output.S3Uri,
                content_type="text/csv")
        }
    )

    custom_model = Model(
        image_uri=f'{uri_base}-inference',
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=sagemaker_role,
        sagemaker_session=pipeline_session,
        entry_point='./steps/predict.py'
    )

    dtt = datetime.now().strftime('%Y%m%d%H%M')
    serverless_inference_config = ServerlessInferenceConfig(
        memory_size_in_mb=6144,
        max_concurrency=1
    )
    create_model_step = ModelStep(
        name=f'GenreDetectorModel{dtt}',
        step_args=custom_model.create(
            serverless_inference_config=serverless_inference_config
        )
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            learning_rate_param,
            batch_size_param,
            epochs_param,
            weight_decay_param,
            threshold_param,
            hidden_layer_1_param,
            hidden_layer_2_param
        ],
        steps=[step_process, step_train, create_model_step]
    )
    pipeline.upsert(role_arn=sagemaker_role)
    pipeline.start()

