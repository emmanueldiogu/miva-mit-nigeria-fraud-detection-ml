FEATURE_COLUMNS = [
    "amount_ngn",
    "velocity_score",
    "spending_deviation_score",
    "user_avg_txn_amt",
    "user_std_txn_amt",
    "user_txn_frequency_24h",
    "txn_count_last_24h",
    "total_amount_last_1h",
    "channel_risk_score",
    "persona_fraud_risk",
    "location_fraud_risk",
]

TARGET_COLUMN = "is_fraud"
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAMPLE_FRAC = 0.05
