# zelestra_aws_hackathon

# ⚡ Solar Panel Efficiency Prediction – Hackathon Submission

This repository contains a machine learning solution for predicting solar panel efficiency. The goal is to build a regression model that predicts the **efficiency** of solar panels based on environmental and operational parameters.

---

## 📂 Files Included

- `source_code1.ipynb`: Main notebook with complete workflow – data preprocessing, model training, evaluation, and submission generation.
- `data_train.csv`: Training dataset with features and target column (`efficiency`).
- `data_test.csv`: Test dataset used to make final predictions.
- `sample_submission.csv`: Sample file indicating correct format for the final submission.
- `submission.csv`: File generated with predicted efficiencies, ready for submission.

---

## 🧠 Problem Statement

Predict the **efficiency** of solar panels using various sensor readings and metadata. Use the provided training data to build a model and predict efficiency values for the test set.

---

## 🧰 Tech Stack

- Python 3
- Jupyter Notebook
- Pandas & NumPy
- Scikit-learn
- XGBoost

---

## 🔄 Workflow

1. **Data Preprocessing**:
   - Drop missing values (`dropna`).
   - Drop non-informative columns (like `id` for model training).
   - Apply **Label Encoding** to categorical columns such as `installation_type`, `string_id`, `error_code`, etc.

2. **Model Training**:
   - Model used: `XGBoostRegressor`
   - Evaluation metrics:
     - `R² Score` for goodness of fit.
     - `RMSLE` (Root Mean Squared Logarithmic Error) to penalize large errors in prediction.

3. **Prediction & Submission**:
   - Generate predictions on `data_test`.
   - Create `submission.csv` with the exact shape **(12000, 2)**:
     - `id`: from `data_test`
     - `efficiency`: predicted value

---

## 📊 Evaluation Code Snippet

```python
from sklearn.metrics import r2_score, mean_squared_log_error
r2 = r2_score(y_train, model.predict(X_train))
rmsle = np.sqrt(mean_squared_log_error(y_train, model.predict(X_train)))
print("R² Score:", r2)
print("RMSLE:", rmsle)
