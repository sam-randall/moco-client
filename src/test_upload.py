
from src.get_fast_models import get_fast_rules
import numpy as np

if __name__ == "__main__":
    dataset_test = np.zeros((10, 2))
    predictions = np.ones(10)
    predictions[:5] = 0
    content = get_fast_rules(dataset_test, predictions)
    print("Content", content)