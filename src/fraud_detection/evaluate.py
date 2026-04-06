from __future__ import annotations

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_classifier(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    return {
        "predictions": y_pred,
        "probabilities": y_proba,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def evaluate_anomaly_detector(model, X_test_scaled, y_test):
    raw_scores = model.decision_function(X_test_scaled)
    anomaly_labels = model.predict(X_test_scaled)
    y_pred = (anomaly_labels == -1).astype(int)
    return {
        "predictions": y_pred,
        "scores": raw_scores,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "roc_auc": roc_auc_score(y_test, -raw_scores),
    }
