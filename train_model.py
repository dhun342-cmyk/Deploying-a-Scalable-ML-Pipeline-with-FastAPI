import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# load the census.csv data
data_path = "data/census.csv"
data = pd.read_csv(data_path)

# split the data into train and test sets
train, test = train_test_split(data, test_size=0.2, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# process training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# process test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# train model
model = train_model(X_train, y_train)

# make sure model directory exists
os.makedirs("model", exist_ok=True)

# save model and encoder
model_path = "model/model.pkl"
encoder_path = "model/encoder.pkl"
save_model(model, model_path)
save_model(encoder, encoder_path)

# load model
model = load_model(model_path)

# run inference
preds = inference(model, X_test)

# print metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

with open("slice_output.txt", "w") as f:
    f.write("")


# compute performance on slices
for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            test,
            col,
            slicevalue,
            cat_features,
            "salary",
            encoder,
            lb,
            model,
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(
                f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}",
                file=f,
            )
