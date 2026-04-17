from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    model_path = artifacts_dir / "iris_model.joblib"
    labels_path = artifacts_dir / "class_names.joblib"
    joblib.dump(model, model_path)
    joblib.dump(list(iris.target_names), labels_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved class labels to: {labels_path}")
    print(
        "\nSample prediction payload:\n"
        "{\n"
        '  "sepal_length": 5.1,\n'
        '  "sepal_width": 3.5,\n'
        '  "petal_length": 1.4,\n'
        '  "petal_width": 0.2\n'
        "}"
    )


if __name__ == "__main__":
    main()
