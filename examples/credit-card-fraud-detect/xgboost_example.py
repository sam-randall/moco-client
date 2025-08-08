import pandas as pd
import xgboost as xgb
import time
from onnxmltools.convert import convert_xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import onnxruntime as rt
import numpy as np
from src.early_exit_model import EarlyExitModel

from onnxmltools.convert.common.data_types import FloatTensorType

def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df[[col for col in df.columns if col.startswith('V')]].to_numpy()
    y = df['Class'].to_numpy()
    return X, y

def main():

    # Download from: 
    dataset_csv_path = '/Users/samrandall/Desktop/creditcard.csv'

    X, y = load_dataset(dataset_csv_path)

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state = 3)

    print("Dataset Size", "X =", train_x.shape, "y = ", train_y.shape)

    n_estimators = 256
    bst = xgb.XGBClassifier(n_estimators = n_estimators)
    bst.fit(train_x, train_y)

    predictions = bst.predict(train_x)

    test_predictions = bst.predict_proba(test_x) 
    p, r, t = precision_recall_curve(test_y, test_predictions[:, 1])

    # Define initial input type
    initial_type = [('input', FloatTensorType([None, X.shape[1]]))]

    # Convert the model
    
    onnx_model = convert_xgboost(bst, initial_types=initial_type)

    # Save to file
    with open("xgb_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Create inference session
    sess = rt.InferenceSession("xgb_model.onnx", providers=['CPUExecutionProvider'])

    # Prepare input

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = X.astype(np.float32)

    pred_onnx = sess.run([output_name], {input_name: input_data})[0]
    # Run inference
    start = time.time()
    pred_onnx = sess.run([output_name], {input_name: input_data})[0]
    end = time.time()
    print("ONNX", end - start)

    start = time.time()
    _ = bst.predict(X)
    end = time.time()
    print("Baseline Time", end - start)

    m = EarlyExitModel(bst)
    predictions = bst.predict(train_x)

    summary = m.compute_short_circuit_rules(train_x, predictions, 1e-7)

    # Disactivate the rule associated with the fraud class.
    m.active_rules[1] = False
    start = time.time()
    m.predict(X)
    end = time.time()
    print("Experimental", end - start)


if __name__ == "__main__":
    main()