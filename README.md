# AI Utilities

This repository contains a simple training utility `mlflow_random_forest.py` that trains a `RandomForestClassifier` on a CSV dataset and logs metrics and artifacts to MLflow.

```bash
python mlflow_random_forest.py data.csv --target label \
    --test-size 0.25 --n-estimators 200
```

The command above expects `pandas`, `scikit-learn`, and `mlflow` to be installed.
