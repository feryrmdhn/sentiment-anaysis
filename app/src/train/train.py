import os
import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker import image_uris, Session
from app.utils.utils import get_sagemaker_client
from dotenv import load_dotenv

load_dotenv()

bucket_name = os.getenv('AWS_BUCKET_NAME')
role = os.getenv('AWS_SAGEMAKER_ROLE_ARN')
aws_region = os.getenv('AWS_REGION')

prefix = 'linear-learner-asset'

if not all([bucket_name, role]):
    raise ValueError("AWS_BUCKET_NAME dan AWS_SAGEMAKER_ROLE_ARN harus diisi di .env!")

region = aws_region
boto3.setup_default_session(region_name=region)
session = Session(boto3.session.Session(region_name=region))

# Dapatkan image uri Linear Learner
container = image_uris.retrieve(
    framework='linear-learner',
    region=region,
    version='1.5-1'
)

# Inisialisasi estimator SageMaker
estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path=f's3://{bucket_name}/{prefix}/model/output',
    sagemaker_session=session,
    hyperparameters={
        'predictor_type': 'binary_classifier',
        'optimizer': 'auto',
        'mini_batch_size': 16
    }
)

# Channel input: training dan validation (test)
train_data_uri = f's3://{bucket_name}/{prefix}/train/train.csv'
test_data_uri = f's3://{bucket_name}/{prefix}/test/test.csv'

train_data = TrainingInput(train_data_uri, content_type='text/csv')
test_data  = TrainingInput(test_data_uri, content_type='text/csv')

def register_model(estimator, model_name, role, container, region):
    client = get_sagemaker_client()

    response = client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            'Image': container,
            'ModelDataUrl': estimator.model_data
        }
    )
    print(f"Model berhasil di-register!")
    return response

if __name__ == '__main__':
    estimator.fit({'train': train_data, 'validation': test_data})

    print("Register model")

    register_model(
        estimator,
        'linear-learner-sentiment-model',
        role,
        container,
        region
    )
