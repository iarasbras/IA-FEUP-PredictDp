# Student Dropout Predictor

This project is a proof-of-concept machine learning system for predicting student dropout risk.

## Goal

The goal is to estimate whether a student has a Low, Medium, or High dropout risk based only on school-based indicators.

## Inputs

- Attendance percentage
- Average grade
- Failed courses
- Completed courses
- Assignment completion percentage
- District of origin
- High school average
- Moodle / platform logins per week

## Output

- Dropout risk: Low / Medium / High

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Streamlit
- joblib
- matplotlib

## How to Run

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate