import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.neural_network import MLPClassifier
import time
import numpy as np
from early_exit_model import EarlyExitModel

def main():
    X, y = load_breast_cancer(return_X_y=True)
    mlp = MLPClassifier(hidden_layer_sizes=16)
    mlp.fit(X, y)

    m = EarlyExitModel(mlp)
    predictions = mlp.predict(X)
    np.save('iris.npy', X)
    np.save('iris_predictions.npy', predictions)
    print(predictions.shape)
    summary = m.get_fast_rules('iris.npy', 'iris_predictions.npy')
    print(summary)

    for i in range(5):
        _ = m.predict(X)
    start = time.time()
    for i in range(5):
        new_predictions = m.predict(X)
    end = time.time()
    print("Experimental", end - start)

    start = time.time()
    for i in range(5):
        out = mlp.predict(X)
    end = time.time()
    print("Original", end - start)
    adherence = (new_predictions == out).mean()
    print("Adherence:", adherence)


if __name__ == "__main__":
    main()

