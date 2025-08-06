

import numpy as np
import json
import requests
from urllib.parse import urlparse
import pandas as pd
from schema import Rule
from sklearn.neural_network import MLPClassifier

# TODO: Move to .env
IS_DEV = True
URL = f'{"127.0.0.1:8000" if IS_DEV else "production-url"}'

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def get_rule(dataset_path: str, prediction_path: str, epsilon: float):

    r = requests.post(f"http://{URL}/get-signed-url")
    data = r.json()

    dataset_url = data['dataset_url']
    prediction_url = data['prediction_url']

    headers = {
        'Content-Type': 'application/x-npy'
    }
    with open(dataset_path, 'rb') as f:
        response = requests.put(dataset_url, data=f, headers=headers)

    if response.status_code == 200:
        pass
    else:
        response.raise_for_status()

    with open(prediction_path, 'rb') as f:
        response = requests.put(prediction_url, data=f, headers=headers)

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
                                                                    'epsilon': epsilon})

    return r


class EarlyExitModel:
    def __init__(self, model):
        self.model = model
        self.membership_rules = []
        self.membership_values = []
        self.active_rules = []

    def get_fast_rules(self, dataset_path: str, prediction_path: str, epsilon: float = 1e-10):
        r = get_rule(dataset_path, prediction_path, epsilon)

        if r.status_code == 200:
            d = json.loads(r.content)
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
        else:
            return None

    def predict(self, x: np.ndarray):

        out = np.zeros(x.shape[0])
        needs_eval = np.arange(x.shape[0])
        for i in range(len(self.membership_rules)):
            if self.active_rules[i]:
                rule = self.membership_rules[i]
                W = np.array(rule.coef)
                b = np.array(rule.intercept)
                t = np.array(rule.threshold)
                p = (sigmoid(x[needs_eval].dot(W.T) + b) >= t)
                p = p[:, 0]

                out[needs_eval[p]] = self.membership_values[i]
                needs_eval = needs_eval[~p]


        if needs_eval.sum() > 0:
            if isinstance(self.model, MLPClassifier):
                out[needs_eval] = self.model.predict(x[needs_eval])
            else:
                raise NotImplementedError(f"Model instance {type(self.model)} not implemented.")

        return out