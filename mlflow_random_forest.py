import argparse
import pickle


def main(
    csv_path: str,
    target_col: str,
    model_output: str,
    test_size: float = 0.2,
    n_estimators: int = 100,
) -> None:
    """Train a random forest and log results to MLflow.

    Parameters
    ----------
    csv_path : str
        Path to the CSV dataset.
    target_col : str
        Column name of the target variable.
    model_output : str
        Path where the trained model pickle will be saved.
    test_size : float, optional
        Fraction of data used for testing (default is 0.2).
    n_estimators : int, optional
        Number of trees in the random forest (default is 100).
    """

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import mlflow
    import mlflow.sklearn

    mlflow.sklearn.autolog()

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    mlflow.log_param("test_size", test_size)
    mlflow.log_param("n_estimators", n_estimators)

    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, artifact_path="model")

    with open(model_output, "wb") as f:
        pickle.dump(clf, f)
    mlflow.log_artifact(model_output)

    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RandomForestClassifier")
    parser.add_argument("csv_path", help="Path to CSV dataset")
    parser.add_argument(
        "--target", default="target", help="Name of the target column in the CSV"
    )
    parser.add_argument(
        "--model-output",
        default="best_model.pkl",
        help="Path to save the best model pickle",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the random forest",
    )
    args = parser.parse_args()

    import mlflow

    mlflow.set_experiment("random_forest_experiment")
    with mlflow.start_run():
        main(
            args.csv_path,
            args.target,
            args.model_output,
            test_size=args.test_size,
            n_estimators=args.n_estimators,
        )
