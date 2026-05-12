import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


DATA_PATH = "data/student_dropout_data.csv"
MODEL_PATH = "models/dropout_model.pkl"


def main():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["dropout_risk"])
    y = df["dropout_risk"]

    numeric_features = [
        "attendance_percentage",
        "average_grade",
        "failed_courses",
        "completed_courses",
        "assignment_completion_percentage",
        "high_school_average",
        "platform_logins_per_week",
    ]

    categorical_features = ["district_of_origin"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            max_iter=1000,
            random_state=42,
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    best_model = None
    best_name = None
    best_accuracy = 0
    results = []

    print("\nMODEL COMPARISON\n")

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"{name}")
        print(f"Accuracy: {accuracy:.3f}")
        print(classification_report(y_test, predictions, zero_division=0))
        print("-" * 50)

        results.append({
            "model": name,
            "accuracy": round(accuracy, 3),
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline
            best_name = name

    joblib.dump(best_model, MODEL_PATH)

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/model_results.csv", index=False)

    print(f"\nBest model: {best_name}")
    print(f"Best accuracy: {best_accuracy:.3f}")
    print(f"Saved model to: {MODEL_PATH}")
    print("Saved model comparison to: models/model_results.csv")

    final_predictions = best_model.predict(X_test)
    cm = confusion_matrix(y_test, final_predictions, labels=["Low", "Medium", "High"])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low", "Medium", "High"],
    )

    display.plot()
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")

    print("Saved confusion matrix to: models/confusion_matrix.png")


if __name__ == "__main__":
    main()