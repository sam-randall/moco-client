
import numpy as np
import requests
import io
import os
from pydantic import BaseModel
from typing import Any, List, Optional
from datetime import datetime
class RulesRequest(BaseModel):
    data: List
    predictions: List
    epsilon: float

class Rule(BaseModel):
    coef: List
    intercept: List
    threshold: float
    quad_coef: Optional[List] = None

class RulesResponse(BaseModel):
    rules: List[Rule]
    rule_values: List[Any]
    rule_summary: List

# TODO: Move to .env
IS_DEV = True
URL = f'{"127.0.0.1:8000" if IS_DEV else "production-url"}'

def array_to_bytes(array: np.ndarray):
    dataset_io = io.BytesIO()
    np.save(dataset_io, array)
    dataset_io.seek(0)
    return dataset_io

from botocore.exceptions import ClientError

def generate_presigned_url(s3_client, client_method, method_parameters, expires_in):
    """
    Generate a presigned Amazon S3 URL that can be used to perform an action.
    
    :param s3_client: A Boto3 Amazon S3 client.
    :param client_method: The name of the client method that the URL performs.
    :param method_parameters: The parameters of the specified client method.
    :param expires_in: The number of seconds the presigned URL is valid for.
    :return: The presigned URL.
    """
    try:
        url = s3_client.generate_presigned_url(
            ClientMethod=client_method,
            Params=method_parameters,
            ExpiresIn=expires_in
        )
    except ClientError:
        print(f"Couldn't get a presigned URL for client method '{client_method}'.")
        raise
    return url

def get_fast_rules_s3(dataset: np.ndarray, predictions: np.ndarray):

    # dataset_io = array_to_bytes(dataset)
    f_name = f'dataset_{datetime.now()}.npy'
    np.save(f_name, dataset)
    import boto3

    key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    
    # with open(object_name, 'rb') as f:
    #     files = {'file': (object_name, f)}
    #     http_response = requests.post(response['url'], data=response['fields'], files=files)

    s3_client = boto3.client("s3")

    # s3_client = boto3.client("s3")
    BUCKET = 'moco-main'
    key = 'dataset.npy'
    
    # The presigned URL is specified to expire in 1000 seconds
    url = generate_presigned_url(
        s3_client, 
        "put_object", 
        {"Bucket": BUCKET, "Key": key}, 
        1000
    )

    with open('dataset.npy', 'rb') as f:
        requests.put(url, files = {'file': ('dataset.npy', f)})

    print(url)
    
    # The presigned URL is specified to expire in 1000 seconds
    # url = generate_presigned_url(
    #     s3_client, 
    #     "put_object", 
    #     {"Bucket": args.bucket, "Key": args.key}, 
    #     1000
    # )

    BUCKET = 'moco-main'
    key = 'dataset.npy'
    # f_name = 'dataset.npy'

    s3 = session.resource('s3')
    upload_url = s3.create_presigned_post(BUCKET, f_name, client_method='put_object', expiration=600)

    with open(f_name, 'rb') as f:
        files = {'file': (f_name, f)}
        http_response = requests.post(upload_url['url'], data=upload_url['fields'], files=files)
        print(http_response.status_code)
    # Filename - File to upload
    # Bucket - Bucket to upload to (the top level directory under AWS S3)
    # Key - S3 object name (can contain subdirectories). If not specified then file_name is used
    s3.meta.client.upload_file(Filename='dataset.npy', Bucket='moco-main', Key='dataset.npy')

    # files = {'file': ('dataset.npy', dataset_io, 'application/octet-stream')}

    # response = requests.post(url=f'http://{URL}/upload', files=files)

    # predictions_io = array_to_bytes(predictions)
    
    # files = {'file': ('predictions.npy', predictions_io, 'application/octet-stream')}
    # response = requests.post(url = f'http://{URL}/upload', files=files)

    # response = requests.post(url = f'http://{URL}/get-rules-from-files', params = {'dataset_path': 'dataset.npy', 'predictions_path': 'predictions.npy', 'epsilon': 1e-6})

    # return RulesResponse.model_validate_json(response.content)



def get_fast_rules(dataset: np.ndarray, predictions: np.ndarray):

    dataset_io = array_to_bytes(dataset)

    files = {'file': ('dataset.npy', dataset_io, 'application/octet-stream')}

    response = requests.post(url=f'http://{URL}/upload', files=files)

    predictions_io = array_to_bytes(predictions)
    
    files = {'file': ('predictions.npy', predictions_io, 'application/octet-stream')}
    response = requests.post(url = f'http://{URL}/upload', files=files)

    response = requests.post(url = f'http://{URL}/get-rules-from-files', params = {'dataset_path': 'dataset.npy', 'predictions_path': 'predictions.npy', 'epsilon': 1e-6})

    return RulesResponse.model_validate_json(response.content)


def get_fast_rules_files(dataset_name, prediction_path):

    # dataset_io = array_to_bytes(dataset)
    print(dataset_name, prediction_path)

    with open(dataset_name, 'rb') as f, open(prediction_path, 'rb') as g:
        response = requests.post(url=f'http://{URL}/upload_data', data = f)
        print("done.")
        response.raise_for_status()

    # predictions_io = array_to_bytes(predictions)
    
    with open(dataset_name, 'rb') as f, open(prediction_path, 'rb') as g:
        files = {'file': ('predictions.npy', g, 'application/octet-stream')}

        response = requests.post(url = f'http://{URL}/upload', files=files)

    response = requests.post(url = f'http://{URL}/get-rules-from-files', params = {'dataset_path': 'dataset.npy', 'predictions_path': 'predictions.npy', 'epsilon': 1e-6})

    return RulesResponse.model_validate_json(response.content)

    