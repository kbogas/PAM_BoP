import time

import numpy as np
import tudataset_utils
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from utils import set_all_seeds

random_state = 42
set_all_seeds(random_state)

# Load data

train_graphs = tudataset_utils.TUDataset("ZINC_train", small=True).read()
valid_graphs = tudataset_utils.TUDataset("ZINC_val", small=True).read()
test_graphs = tudataset_utils.TUDataset("ZINC_test", small=True).read()


y_train = [t.label_ohe for t in train_graphs]
y_valid = [t.label_ohe for t in valid_graphs]
y_test = [t.label_ohe for t in test_graphs]


time_s = time.time()


pipe = Pipeline(
    [
        (
            "pam",
            tudataset_utils.PAMs_from_ZINC(power=5),
        ),
        ("bop", tudataset_utils.BoP()),
        ("clf", LinearSVR(max_iter=3000, random_state=random_state)),
    ]
)

print("\nLoaded data. \n Will generate PAMs and fit BoPs")
pipe.fit(train_graphs, y_train)

print(f"\nPredicting...")
y_train_pred = pipe.predict(train_graphs)

y_pred = pipe.predict(test_graphs)

print(f"\nTrain MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred):.4f}")

print(f"\nTime taken: {(time.time() - time_s)/60:.2f} mins")
