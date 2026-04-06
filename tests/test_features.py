import pandas as pd
from fraud_detection.features import select_features, sample_and_split


def test_select_features_removes_missing_rows():
    df = pd.DataFrame({
        "amount_ngn": [1000, 2000, None],
        "velocity_score": [0.1, 0.2, 0.3],
        "is_fraud": [0, 1, 0],
    })
    X, y = select_features(df, ["amount_ngn", "velocity_score"], "is_fraud")
    assert len(X) == 2
    assert len(y) == 2


def test_sample_and_split_returns_non_empty_sets():
    df = pd.DataFrame({
        "amount_ngn": list(range(100)),
        "velocity_score": list(range(100)),
    })
    y = pd.Series([0] * 90 + [1] * 10)
    X_train, X_test, y_train, y_test = sample_and_split(df, y, sample_frac=0.5, test_size=0.2, random_state=42)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
