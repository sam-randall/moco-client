import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.neural_network import MLPClassifier
import time
import numpy as np
from early_exit_model import EarlyExitModel

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

