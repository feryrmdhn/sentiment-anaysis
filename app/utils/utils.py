import os
import boto3
import joblib
from dotenv import load_dotenv

load_dotenv()

bucket_name = os.getenv('AWS_BUCKET_NAME')
aws_region = os.getenv('AWS_REGION')
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

def get_s3_client():
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    return session.client('s3')

def get_sagemaker_client():
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    return session.client('sagemaker')

def upload_to_s3(local_file, bucket, s3_path):
    s3_client = get_s3_client()
    s3_client.upload_file(local_file, bucket, s3_path)
    print(f"Sukses upload {local_file} ke s3://{bucket}/{s3_path}")

def load_vectorizer_from_s3():
    local_vectorizer_filename = 'tfidf_vectorizer.joblib'
    s3_key = 'linear-learner-asset/artifact/tfidf_vectorizer.joblib'

    try:
        s3_client = get_s3_client()
        
        # Cek apakah bucket ada
        s3_client.head_bucket(Bucket=bucket_name)
        
        # Cek apakah object ada
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"Object {s3_key} ditemukan.")
        
        # Download file dari S3
        s3_client.download_file(bucket_name, s3_key, local_vectorizer_filename)
        
        # Load vectorizer
        vectorizer = joblib.load(local_vectorizer_filename)
        print("Vectorizer berhasil di-load!")
        
        os.remove(local_vectorizer_filename)  # Hapus file lokal setelah load
        
        return vectorizer
    except Exception as e:
        raise RuntimeError(f"Gagal load vectorizer dari S3: {e}")