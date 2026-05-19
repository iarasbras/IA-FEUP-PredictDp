# Student Dropout Predictor

A machine learning model to predict student dropout risk (Low, Medium, High) based on school indicators.

## Problem

Predict dropout risk using features: attendance, grades, failed/completed courses, assignment completion, high school average, platform logins, and district.

## Solution

Supervised multiclass classifier comparing 4 algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- Neural Network

The training pipeline compares Logistic Regression, Decision Tree, Random Forest, and Neural Network models. The best-performing model is selected automatically and saved as `models/dropout_model.pkl`.

## Data

- Training: `data/student_dropout_training_data.csv`
- Testing: `data/student_dropout_testing_data.csv` (synthetic, generated)
- Generator: `data/data_generator.py` (enforces grade floor 9.5, course constraints, realistic correlations)

## Evaluation

- 5-fold cross-validation per algorithm
- Holdout test metrics (accuracy, precision, recall, F1)
- Confusion matrix, ROC curves, calibration plots
- Results: `models/`

## Setup & Run Instructions

**1. Create and activate virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Generate synthetic testing data:**

```bash
python data/data_generator.py
```

**3. Train and compare models:**

```bash
python models/train_model.py
```

**4. Run post-training evaluation:**

```bash
python models/evaluate_model.py
```

**5. Run interactive web app demo:**

```bash
streamlit run app/app.py
```
