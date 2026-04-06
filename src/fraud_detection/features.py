from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def select_features(df: pd.DataFrame, feature_columns: list[str], target_column: str):
    data = df[feature_columns + [target_column]].dropna().copy()
    X = data[feature_columns]
    y = data[target_column]
    return X, y


def sample_and_split(X, y, sample_frac=0.05, test_size=0.2, random_state=42):
    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_frac,
        stratify=y,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample,
        y_sample,
        test_size=test_size,
        stratify=y_sample,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled
