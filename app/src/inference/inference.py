import boto3
import os
from dotenv import load_dotenv
import time

load_dotenv()

aws_region = os.getenv('AWS_REGION')
model_name = "linear-learner-sentiment-model"  

instance_type = "ml.m4.xlarge"

def delete_if_exists(client, resource_type, name):
    if resource_type == 'endpoint':
        try:
            client.describe_endpoint(EndpointName=name)
            print(f"Endpoint '{name}' sudah ada, menghapus...")
            client.delete_endpoint(EndpointName=name)

            # Tunggu sampai endpoint benar-benar terhapus
            waiter = client.get_waiter('endpoint_deleted')
            print("Menunggu endpoint terhapus...")
            waiter.wait(EndpointName=name)
            print("Endpoint terhapus.")

        except client.exceptions.ClientError as e:
            if "Could not find endpoint" in str(e):
                pass  # Endpoint tidak ada
            else:
                raise
    elif resource_type == 'endpoint-config':
        try:
            client.describe_endpoint_config(EndpointConfigName=name)
            print(f"Endpoint config '{name}' sudah ada, menghapus...")
            client.delete_endpoint_config(EndpointConfigName=name)
            print("Endpoint config terhapus.")

        except client.exceptions.ClientError as e:
            if "Could not find endpoint configuration" in str(e):
                pass  # Config tidak ada
            else:
                raise

def create_endpoint_config(model_name, region, instance_type):
    client = boto3.client('sagemaker', region_name=region)
    endpoint_config_name = f"{model_name}-endpoint-config"

    # Cek dan hapus config jika sudah ada
    delete_if_exists(client, 'endpoint-config', endpoint_config_name)
    response = client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InstanceType': instance_type,
            'InitialInstanceCount': 1
        }]
    )
    return endpoint_config_name

def create_endpoint(endpoint_config_name, region):
    client = boto3.client('sagemaker', region_name=region)

    # Ganti 'endpoint-config' menjadi 'endpoint'
    endpoint_name = endpoint_config_name.replace('-endpoint-config', '-endpoint')

    # Cek dan hapus endpoint jika sudah ada
    delete_if_exists(client, 'endpoint', endpoint_name)
    response = client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    print("Endpoint dalam proses creating...")

    waiter = client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    print("Endpoint sudah siap digunakan!")
    return endpoint_name

if __name__ == '__main__':
    endpoint_config_name = create_endpoint_config(model_name, aws_region, instance_type)
    create_endpoint(endpoint_config_name, aws_region)
