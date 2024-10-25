import time

import numpy as np
import tudataset_utils
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR

print(f"Running graph regression on AQSL..")
data = load_dataset("graphs-datasets/AQSOL")

y_train = [g["y"][0] for g in data["train"]]
y_valid = [g["y"][0] for g in data["val"]]
y_test = [g["y"][0] for g in data["test"]]
print(f"# Train {len(y_train)}\t# Train {len(y_valid)}\t# Test {len(y_test)}")


train_graphs = [item for item in data["train"]]
valid_graphs = [item for item in data["val"]]
test_graphs = [item for item in data["test"]]


time_s = time.time()

pipe = Pipeline(
    [
        (
            "pam",
            tudataset_utils.PAMs_from_AQSL(power=9),
        ),
        ("hier", tudataset_utils.BoP()),
        ("clf", LinearSVR(max_iter=3000, random_state=42)),
    ]
)

print("\nLoaded data.\nWill generate PAMs and fit BoPs.")
pipe.set_params()


pipe.fit(train_graphs, y_train)
print(f"\nPredicting...")
y_train_pred = pipe.predict(train_graphs)

y_pred = pipe.predict(test_graphs)

print(f"Train MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Time taken: {(time.time() - time_s)/60:.2f} mins")
