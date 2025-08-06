import numpy as np
import requests
import json
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.neural_network import MLPClassifier
import time
from pydantic import BaseModel
from typing import Any, List, Optional

import requests
from urllib.parse import urlparse
import numpy as np

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
        needs_eval = np.ones(x.shape[0], dtype = np.bool_)
        for i in range(len(self.membership_rules)):
            if self.active_rules[i]:
                rule = self.membership_rules[i]
                W = np.array(rule.coef)
                b = np.array(rule.intercept)
                t = np.array(rule.threshold)
                write_mask = needs_eval.copy()
                p = (sigmoid(x[needs_eval].dot(W.T) + b) >= t)

                p = p[:, 0]

                write_mask &= p
                out[write_mask] = self.membership_values[i]

                needs_eval[write_mask] = 0
                break

        if needs_eval.sum() > 0:
            out[needs_eval] = self.model.predict(x[needs_eval])

        return out


def main():
    X, y = load_breast_cancer(return_X_y=True)
    mlp = MLPClassifier()
    mlp.fit(X, y)

    m = EarlyExitModel(mlp)
    predictions = mlp.predict(X)
    np.save('iris.npy', X)
    np.save('iris_predictions.npy', predictions)
    print(predictions.shape)
    summary = m.get_fast_rules('iris.npy', 'iris_predictions.npy')
    print(summary)

    start = time.time()
    new_predictions = m.predict(X)
    end = time.time()
    print(end - start)

    start = time.time()
    out = mlp.predict(X)
    end = time.time()
    print(end - start)
    adherence = (new_predictions == out).mean()
    print("Adherence:", adherence)


if __name__ == "__main__":
    main()

