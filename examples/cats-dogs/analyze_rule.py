import numpy as np
import pandas as pd
from src.schema import RulesResponse
import json
from tqdm import tqdm

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def main():
    embedding_path = 'resnet_cats_dogs_embeddings.npy'
    prediction_path = 'classifications.npy'

    data = np.load(embedding_path)
    p = np.load(prediction_path)
    rules_file = 'rules.json'
    with open(rules_file) as f:
        d = json.load(f)
        d = json.loads(d)
        response = RulesResponse.model_validate(d)
        for rule in response.rules:
            W = np.array(rule.coef)
            b = np.array(rule.intercept)
            t = np.array(rule.threshold)
            B = 1000
            activated_count = 0
            activated_mask = np.zeros(p.shape[0])
            for i in tqdm(range(0, len(data), B)):
                B_ = data[i: i + B].shape[0]
                decision = data[i: i + B].reshape((B_, -1)).dot(W.T) + b
                out = sigmoid(decision) >= t
                activated_mask[i: i + B] = out[:, 0]
                activated_count += out.sum()
            print(pd.Series(p[activated_mask.astype(np.bool_)]).value_counts())
            print("Activated Count", activated_count)
                
            
        
if  __name__ == "__main__":
    main()
