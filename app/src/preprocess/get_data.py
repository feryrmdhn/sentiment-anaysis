import os
import boto3
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from app.utils.utils import get_s3_client

load_dotenv()

bucket_name = os.getenv('AWS_BUCKET_NAME')
dataset = os.getenv('DATASET_NAME')

if not bucket_name:
    raise ValueError("Environment variable AWS_BUCKET_NAME harus di set diawal!")

prefix = 'dataset'
file_key = f"{prefix}/{dataset}"

s3 = get_s3_client()

def get_data_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response['Body'].read().decode('utf-8')  # csv
    df = pd.read_csv(StringIO(body))
    return df

def load_dataframe():
    return get_data_from_s3(bucket_name, file_key)

if __name__ == '__main__':
    df = load_dataframe()
