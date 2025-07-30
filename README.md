# AI Utilities

This repository contains two training utilities for experimenting with MLflow.

- `mlflow_random_forest.py` trains a `RandomForestClassifier` on a CSV dataset and logs metrics and artifacts to MLflow.
- `mlflow_random_forest_cv.py` performs a grid search over `n_estimators` values and logs the best model.

Example usage:

```bash
python mlflow_random_forest.py data.csv --target label \
    --test-size 0.25 --n-estimators 200

python mlflow_random_forest_cv.py data.csv --target label \
    --n-estimators 50 100 200 400
```

The commands above expect `pandas`, `scikit-learn`, and `mlflow` to be installed.

