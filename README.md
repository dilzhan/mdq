# Mastercard Data Quest: Hidden Business Detection

This project was developed for Mastercard Data Quest, a case championship organized by Mastercard and AIESEC Kazakhstan. The task was to study consumer-like and business-like card transactions and detect hidden businesses using consumer cards.

We aggregated transactional data into a card-level dataset and trained several models, including Logistic Regression, KNN, SVM, Random Forest, and XGBoost. Afterwards, the models were combined into an ensemble-like model that averages model scores, identifies the decision threshold based on True Positive Rate (TPR), and estimates model uncertainty for a given data point.

The results of the model were visualized using Streamlit.

As a bonus, we trained a neural network and added a Laplace approximation to make it probabilistic. In the end, we obtained a Gaussian Process-like model with the trained neural network as the mean function and variance estimated using the Laplace approximation.

## Structure

```text
src/          core feature engineering, data, and modeling code
notebooks/    EDA, feature selection, model experiments, final pipeline
dashboard/    Streamlit dashboard for reviewing predictions
graphs/       generated figures
data/         local input data, not tracked by git
models/       local trained model files, not tracked by git
```

## Setup

Create an environment and install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dashboard

Generate `dashboard/dashboard_data.csv` locally, then run:

```bash
streamlit run dashboard/main.py
```

## Notes

Raw data, dashboard exports, and model artifacts are excluded from git.
