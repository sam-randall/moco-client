
import numpy as np
import requests
import io

from pydantic import BaseModel
from typing import Any, List, Optional

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


def get_fast_rules(dataset: np.ndarray, predictions: np.ndarray):

    dataset_io = array_to_bytes(dataset)

    files = {'file': ('dataset.npy', dataset_io, 'application/octet-stream')}

    response = requests.post(url=f'http://{URL}/upload', files=files)

    predictions_io = array_to_bytes(predictions)
    
    files = {'file': ('predictions.npy', predictions_io, 'application/octet-stream')}
    response = requests.post(url = f'http://{URL}/upload', files=files)

    response = requests.post(url = f'http://{URL}/get-rules-from-files', params = {'dataset_path': 'dataset.npy', 'predictions_path': 'predictions.npy', 'epsilon': 1e-6})

    return RulesResponse.model_validate_json(response.content)

    