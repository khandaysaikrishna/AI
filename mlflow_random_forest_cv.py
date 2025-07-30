import argparse
import pickle


def main(
    csv_path: str,
    target_col: str,
    model_output: str,
    n_estimators: list[int] = None,
    test_size: float = 0.2,
    cv: int = 3,
) -> None:
    """Train a random forest with cross-validation and log results to MLflow.

    Parameters
    ----------
    csv_path : str
        Path to the CSV dataset.
    target_col : str
        Column name of the target variable.
    model_output : str
        Path where the trained model pickle will be saved.
    n_estimators : list[int], optional
        List of `n_estimators` values to try in the grid search.
        Defaults to [50, 100, 200].
    test_size : float, optional
        Fraction of data used for testing. Default is 0.2.
    cv : int, optional
        Number of cross-validation folds. Default is 3.
    """

    if n_estimators is None:
        n_estimators = [50, 100, 200]

    import pandas as pd
    from sklearn.model_selection import train_test_split, GridSearchCV
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

    param_grid = {"n_estimators": n_estimators}
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = GridSearchCV(clf, param_grid=param_grid, cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)

    best_clf = search.best_estimator_

    predictions = best_clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    mlflow.log_param("test_size", test_size)
    mlflow.log_param("cv", cv)
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(best_clf, artifact_path="model")

    with open(model_output, "wb") as f:
        pickle.dump(best_clf, f)
    mlflow.log_artifact(model_output)

    print(f"Best n_estimators: {search.best_params_['n_estimators']}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RandomForest with cross-validation logged to MLflow"
    )
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
        "--n-estimators",
        nargs="*",
        type=int,
        default=[50, 100, 200],
        help="Space-separated list of n_estimators values to try",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Number of cross-validation folds",
    )
    args = parser.parse_args()

    import mlflow

    mlflow.set_experiment("random_forest_cv_experiment")
    with mlflow.start_run():
        main(
            args.csv_path,
            args.target,
            args.model_output,
            n_estimators=args.n_estimators,
            test_size=args.test_size,
            cv=args.cv,
        )
