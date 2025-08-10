import numpy as np

import json
import requests
from urllib.parse import urlparse
import pandas as pd
from src.schema import Rule
from torch import nn
import os
from typing import Any, Dict, Optional, Union
import torch

# TODO: Move to .env
IS_DEV = True
URL = f'{"127.0.0.1:8000" if IS_DEV else "production-url"}'

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def get_rule(dataset_path: str, prediction_path: str, epsilon: float, email_address: Optional[str] = None):

    r = requests.post(f"http://{URL}/get-signed-url")
    print("Got signed URLs")
    data = r.json()

    dataset_url = data['dataset_url']
    prediction_url = data['prediction_url']

    headers = {
        'Content-Type': 'application/x-npy'
    }
    print("Putting dataset into s3.")

    with open(dataset_path, 'rb') as f:
        response = requests.put(dataset_url, data=f, headers=headers, stream = True)

    print("wrote dataset to s3.")

    if response.status_code == 200:
        pass
    else:
        response.raise_for_status()

    with open(prediction_path, 'rb') as f:
        response = requests.put(prediction_url, data=f, headers=headers, stream = True)

    print("wrote predictions to s3.")
    dataset_remote_path = urlparse(dataset_url).path
    prediction_remote_path = urlparse(prediction_url).path

    if response.status_code == 200:
        pass
    else:
        response.raise_for_status()

    dataset_remote_path = dataset_remote_path.lstrip('/')
    prediction_remote_path = prediction_remote_path.lstrip('/')

    r = requests.post(f"http://{URL}/get-s3-files", params={'dataset_s3_key': dataset_remote_path,
                                                                    'predictions_s3_key': prediction_remote_path,
                                                                    'epsilon': epsilon, 'email_address': email_address})
    print(r.content)
    return r


class EarlyExitModel:
    def __init__(self, model):
        self.model = model
        self.membership_rules = []
        self.membership_values = []
        self.active_rules = []
        self.decision_function = lambda x: x
        self.default_path = self.model

    def compute_short_circuit_rules(self, dataset: np.ndarray, predictions: np.ndarray, epsilon: float, email_address: Optional[str] = None):
        assert isinstance(dataset, np.ndarray)
        assert isinstance(predictions, np.ndarray)
        os.makedirs('tmp', exist_ok=True)
        print("Saving...", dataset.shape, predictions.shape)
        np.save('tmp/dataset.npy', dataset)
        np.save('tmp/predictions.npy', predictions)
        return self.run_fast_rule_job('tmp/dataset.npy', 'tmp/predictions.npy', epsilon, email_address)


    def apply_rules_from_json_string(self, response_as_dict: Dict[str, Any]):
        d = response_as_dict
        rules = d['rules']
        rule_values = d['rule_values']
        rule_summary = d.get('rule_summary', None)

        out = [Rule.model_validate(r) for r in rules]
        self.membership_rules = out
        self.membership_values = rule_values
        self.active_rules = [True] * len(out)
        if rule_summary is not None:
                out = pd.DataFrame(rule_summary)
                return out
    
    def run_fast_rule_job(self, dataset_path: str, prediction_path: str, epsilon: float, email_address: str):
        r = get_rule(dataset_path, prediction_path, epsilon, email_address)
        r.raise_for_status()
        return r

    def predict(self, x: Union[np.ndarray, torch.Tensor]):
        data_library_type = 'np' if isinstance(x, np.ndarray) else 'torch'
        with torch.no_grad():
            x = self.decision_function(x)
        out = np.zeros(x.shape[0])
        needs_eval = np.arange(x.shape[0])
        for i in range(len(self.membership_rules)):
            if self.active_rules[i]:
                rule = self.membership_rules[i]
                W = np.array(rule.coef) if data_library_type == 'np' else torch.Tensor(rule.coef)
                b = np.array(rule.intercept) if data_library_type == 'np' else torch.Tensor(rule.intercept)
                t = rule.threshold # np.array(rule.threshold) if data_library_type == 'np' else torch.Tensor(rule.threshold)
                N = x.shape[0]
                x_ = x.reshape((N, -1))
                xw = x_[needs_eval].dot(W.T) if data_library_type == 'np' else torch.matmul(x_[needs_eval], W.T)
                lin = xw + b
                p = (sigmoid(lin) >= t)
                p = p[:, 0]

                out[needs_eval[p]] = self.membership_values[i]
                needs_eval = needs_eval[~p]

        if needs_eval.sum() > 0:
            try:
                out[needs_eval] = self.default_path(x[needs_eval]).argmax(axis = 1)
            except Exception as e:
                out[needs_eval] = self.default_path(x[needs_eval]).argmax(axis = 1)
        return out
