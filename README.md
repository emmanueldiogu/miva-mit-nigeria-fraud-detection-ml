# Nigeria Fraud Detection ML

A GitHub-ready seminar project for **data-driven fraud detection in digital banking in Nigeria**.

This repository is structured so you can:
- work locally in **VS Code**,
- run experiments in a notebook,
- keep reusable code inside `src/`,
- run tests with `pytest`, and
- later attach results to your seminar paper.

## Project structure

```text
nigeria-fraud-ml/
├── data/
│   ├── raw/                # original datasets
│   └── processed/          # cleaned/model-ready data
├── notebooks/
│   └── 01_model_experiments.ipynb
├── src/
│   └── fraud_detection/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── features.py
│       ├── train.py
│       ├── evaluate.py
│       └── utils.py
├── tests/
│   ├── test_features.py
│   └── test_train.py
├── models/
├── reports/
├── figures/
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

3. Run tests:

```bash
pytest -q
```

4. Start the notebook in VS Code and open:

- `notebooks/01_model_experiments.ipynb`

## Suggested workflow

- Put downloaded dataset files in `data/raw/`.
- Use notebook cells for exploration.
- Move reusable logic into `src/fraud_detection/`.
- Save trained models in `models/`.
- Save charts/tables for the paper in `figures/` and `reports/`.

## Notes

- The dataset source is the Nigerian Financial Transactions and Fraud Detection Dataset on Hugging Face, which is described as a Nigeria-specific fraud dataset with millions of synthetic transactions and engineered fraud-related features (huggingface)[https://huggingface.co/datasets/electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset].
- You can also use the Kaggle NIBSS-style synthetic dataset as an alternative source (kaggle)[https://www.kaggle.com/datasets/hendurhance/nibsss-fraud-dataset/code].
