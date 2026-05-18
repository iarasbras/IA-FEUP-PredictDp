import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer
from sklearn.calibration import calibration_curve


def main():
    MODEL_PATH = "models/dropout_model.pkl"
    TEST_PATH = "data/student_dropout_testing_data.csv"
    OUT_DIR = "models"

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH)

    X = df.drop(columns=["dropout_risk"])
    y = df["dropout_risk"]

    y_pred = model.predict(X)
    y_proba = None
    try:
        y_proba = model.predict_proba(X)
    except Exception:
        pass

    report = classification_report(y, y_pred, output_dict=True)
    with open(f"{OUT_DIR}/eval_classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    cm_df.to_csv(f"{OUT_DIR}/eval_confusion_matrix.csv")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(model.classes_)))
    ax.set_yticks(np.arange(len(model.classes_)))
    ax.set_xticklabels(model.classes_)
    ax.set_yticklabels(model.classes_)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/confusion_matrix_eval.png", dpi=150)
    plt.close(fig)

    # ROC AUC (one-vs-rest) and calibration if probabilities available
    if y_proba is not None:
        lb = LabelBinarizer()
        Yb = lb.fit_transform(y)
        # If binary, LabelBinarizer returns single column — expand to 2 columns
        if Yb.ndim == 1:
            Yb = np.vstack([1 - Yb, Yb]).T

        # ROC AUC per class
        roc_scores = {}
        for i, cls in enumerate(lb.classes_):
            try:
                roc = roc_auc_score(Yb[:, i], y_proba[:, i])
            except Exception:
                roc = None
            roc_scores[str(cls)] = roc
        with open(f"{OUT_DIR}/eval_roc_auc.json", "w") as f:
            json.dump(roc_scores, f, indent=2)

        # Plot ROC curves
        plt.figure(figsize=(6, 5))
        for i, cls in enumerate(lb.classes_):
            try:
                from sklearn.metrics import roc_curve

                fpr, tpr, _ = roc_curve(Yb[:, i], y_proba[:, i])
                plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_scores[str(cls)]:.2f})")
            except Exception:
                continue
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (one-vs-rest)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/roc_eval.png", dpi=150)
        plt.close()

        # Calibration curve for each class (one-vs-rest)
        plt.figure(figsize=(6, 5))
        for i, cls in enumerate(lb.classes_):
            try:
                prob_pos = y_proba[:, i]
                frac_pos, mean_pred = calibration_curve(Yb[:, i], prob_pos, n_bins=10)
                plt.plot(mean_pred, frac_pos, marker="o", label=str(cls))
            except Exception:
                continue
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration plot (one-vs-rest)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/calibration_eval.png", dpi=150)
        plt.close()

    print("Evaluation complete. Outputs saved to models/ (confusion matrix, plots, JSON)")


if __name__ == "__main__":
    main()
