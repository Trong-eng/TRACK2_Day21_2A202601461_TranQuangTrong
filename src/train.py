import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Dung SQLite backend da su dung o Buoc 1 neu runner chua dat URI.
# MLFLOW_TRACKING_URI tu moi truong van duoc uu tien khi can backend khac.
if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Nguong mentor chap nhan cho ket qua cua bo du lieu nay.
EVAL_THRESHOLD = 0.68


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua model_type va cac sieu tham so cua mo hinh.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    # Doc du lieu huan luyen va danh gia.
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Tach dac trung (X) va nhan (y).
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: theo doi phan phoi nhan tren tap train, khong dung eval.
    class_distribution = {
        str(label): float((y_train == label).mean())
        for label in (0, 1, 2)
    }
    distribution_warnings = [
        f"Class {label} chiem {class_distribution[str(label)]:.2%}, duoi 10%"
        for label in (0, 1, 2)
        if class_distribution[str(label)] < 0.10
    ]
    for warning in distribution_warnings:
        print(f"WARNING: {warning}")

    # Mac dinh dung Random Forest de tuong thich voi cac loi goi cu.
    model_params = params.copy()
    model_type = model_params.pop("model_type", "random_forest")
    logged_params = {"model_type": model_type, **model_params}

    if model_type == "random_forest":
        model = RandomForestClassifier(**model_params, random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(**model_params, random_state=42)
    elif model_type == "extra_trees":
        model = ExtraTreesClassifier(**model_params, random_state=42)
    elif model_type == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(**model_params, random_state=42)
    elif model_type == "voting_ensemble":
        n_estimators = model_params.pop("n_estimators", 800)
        weights = model_params.pop("weights", [1, 1])
        voting = model_params.pop("voting", "soft")
        if model_params:
            unsupported = ", ".join(sorted(model_params))
            raise ValueError(f"Unsupported voting_ensemble params: {unsupported}")

        model = VotingClassifier(
            estimators=[
                (
                    "random_forest",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=None,
                        min_samples_split=2,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
                (
                    "extra_trees",
                    ExtraTreesClassifier(
                        n_estimators=n_estimators,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ],
            voting=voting,
            weights=weights,
            n_jobs=1,
        )
    elif model_type == "stacking_ensemble":
        n_estimators = model_params.pop("n_estimators", 500)
        cv = model_params.pop("cv", 3)
        passthrough = model_params.pop("passthrough", False)
        if model_params:
            unsupported = ", ".join(sorted(model_params))
            raise ValueError(f"Unsupported stacking_ensemble params: {unsupported}")

        model = StackingClassifier(
            estimators=[
                (
                    "random_forest",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=None,
                        min_samples_split=2,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
                (
                    "extra_trees",
                    ExtraTreesClassifier(
                        n_estimators=n_estimators,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
                (
                    "hist_gradient_boosting",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_iter=250,
                        max_leaf_nodes=31,
                        min_samples_leaf=10,
                        l2_regularization=0.1,
                        random_state=42,
                    ),
                ),
            ],
            final_estimator=make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=42),
            ),
            cv=cv,
            stack_method="predict_proba",
            passthrough=passthrough,
            n_jobs=1,
        )
    elif model_type == "calibrated_extra_trees":
        n_estimators = model_params.pop("n_estimators", 500)
        method = model_params.pop("method", "sigmoid")
        cv = model_params.pop("cv", 5)
        if model_params:
            unsupported = ", ".join(sorted(model_params))
            raise ValueError(f"Unsupported calibrated_extra_trees params: {unsupported}")

        base_model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=1,
        )
        model = CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv=cv,
            n_jobs=1,
            ensemble=True,
        )
    else:
        raise ValueError(
            "Unsupported model_type: "
            f"{model_type}. Expected 'random_forest', 'gradient_boosting', "
            "'extra_trees', 'hist_gradient_boosting', 'voting_ensemble' "
            "'stacking_ensemble' or 'calibrated_extra_trees'."
        )

    with mlflow.start_run():

        # Ghi nhan loai mo hinh va cac sieu tham so.
        mlflow.log_params(logged_params)

        # Huan luyen mo hinh. random_state=42 giup ket qua co the tai lap.
        model.fit(X_train, y_train)

        # Du doan tren tap danh gia va tinh chi so.
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # Bonus 3: confusion matrix va precision/recall cho tung lop.
        labels = [0, 1, 2]
        matrix = confusion_matrix(y_eval, preds, labels=labels)
        precision, recall, _, _ = precision_recall_fscore_support(
            y_eval,
            preds,
            labels=labels,
            average=None,
            zero_division=0,
        )

        # Ghi nhan chi so va model vao MLflow.
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # In ket qua ra man hinh.
        print(f"Model: {model_type} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Luu metrics ra file de CI/CD doc o Buoc 2.
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump(
                {
                    "model_type": model_type,
                    "accuracy": acc,
                    "f1_score": f1,
                    "label_distribution": class_distribution,
                    "distribution_warnings": distribution_warnings,
                },
                f,
                indent=2,
            )

        report_lines = [
            f"Model type: {model_type}",
            f"Accuracy: {acc:.6f}",
            f"Weighted F1-score: {f1:.6f}",
            "",
            "Confusion matrix (rows=true, columns=predicted; labels=0,1,2):",
        ]
        report_lines.extend(" ".join(str(int(value)) for value in row) for row in matrix)
        report_lines.extend(["", "Per-class metrics:"])
        report_lines.extend(
            f"Class {label}: precision={precision[index]:.6f}, "
            f"recall={recall[index]:.6f}"
            for index, label in enumerate(labels)
        )
        report_lines.extend(["", "Training label distribution:"])
        report_lines.extend(
            f"Class {label}: {class_distribution[str(label)]:.2%}"
            for label in labels
        )
        if distribution_warnings:
            report_lines.extend(["", "Warnings:", *distribution_warnings])
        with open("outputs/report.txt", "w") as f:
            f.write("\n".join(report_lines) + "\n")

        # Luu mo hinh de dung lai o cac Buoc 2 va 3.
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # Tra ve accuracy de cac ham goi train() co the su dung ket qua.
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
