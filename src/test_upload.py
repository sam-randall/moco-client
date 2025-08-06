
from src.get_fast_models import get_fast_rules, get_fast_rules_files
import numpy as np
from datetime import datetime

if __name__ == "__main__":
    dataset_test = np.zeros((10, 2))
    predictions = np.ones(10)
    predictions[:5] = 0
    content = get_fast_rules(dataset_test, predictions)
    print("Content", content)

    dataset_test = np.zeros((25000, 250, 120))
    predictions = np.ones(25000)
    predictions[:5] = 0
    dataset_path = f'dataset_{datetime.now()}.npy'
    predictions_path = f'predictions_{datetime.now()}.npy'
    np.save(dataset_path, dataset_test)
    np.save(predictions_path, predictions)
    content = get_fast_rules_files(dataset_path, predictions_path)
    print("Content", content)