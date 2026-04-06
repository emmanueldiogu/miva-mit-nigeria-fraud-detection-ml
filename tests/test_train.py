import numpy as np
from fraud_detection.train import train_svm, train_isolation_forest


def test_train_svm_fits_model():
    X = np.array([[0.1, 1.0], [0.2, 0.9], [5.0, 5.1], [4.9, 5.2]])
    y = np.array([0, 0, 1, 1])
    model = train_svm(X, y)
    preds = model.predict(X)
    assert len(preds) == 4


def test_train_isolation_forest_fits_model():
    X = np.array([[0.1, 1.0], [0.2, 0.9], [0.15, 1.1], [5.0, 5.1]])
    y = np.array([0, 0, 0, 1])
    model = train_isolation_forest(X, y, contamination=0.25)
    preds = model.predict(X)
    assert len(preds) == 4
