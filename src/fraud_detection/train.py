from __future__ import annotations

from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC


def train_svm(X_train_scaled, y_train, random_state=42):
    model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)
    model.fit(X_train_scaled, y_train)
    return model


def train_isolation_forest(X_train_scaled, y_train, contamination=0.01, random_state=42):
    X_train_normal = X_train_scaled[y_train == 0]
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X_train_normal)
    return model
