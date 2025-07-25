import numpy as np
import requests
import json
import pandas as pd

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



class EarlyExitModel:
    def __init__(self, model):
        self.model = model
        self.membership_rules = []
        self.membership_values = []

    def get_fast_rules(self, dataset: np.ndarray, predictions: np.ndarray):
        dataset = dataset.tolist()
        predictions = predictions.tolist()
        d = {
            'data': dataset,
            'predictions': predictions,
            'epsilon': 1e-14
        }

        r = requests.post(f"http://{URL}/get-rules", json = d)

        if r.status_code == 200:
            d = json.loads(r.content)
            rules = d['rules']
            rule_values = d['rule_values']
            rule_summary = d.get('rule_summary', None)

            out = [Rule.model_validate(r) for r in rules]
            self.membership_rules = out
            self.membership_values = rule_values
            if rule_summary is not None:
                out = pd.DataFrame(rule_summary)
                return out
        else:
            return None

    def predict(self, x: np.ndarray):
        pass

def main():
    m = EarlyExitModel(None)
    data1 = np.random.uniform(low = 0, high = 0.1, size = (20, 2))
    data2 = np.random.uniform(low = 3, high = 3.5, size = (20, 2))
    stack = np.vstack([data1, data2])

    predictions = np.zeros(40)
    predictions[20:40] = 1
    summary = m.get_fast_rules(stack, predictions)
    # print(m.membership_rules)
    # print(m.membership_values)
    print(summary)


if __name__ == "__main__":
    main()

