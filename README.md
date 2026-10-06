# 🩺 SugarSense

## Early Diabetes Risk Screening with Supervised Machine Learning

SugarSense is a machine learning-based diabetes risk screening application.

The project uses clinical measurements to estimate the probability of diabetes and provides an interpretable screening result through a Streamlit web application.

> ⚠️ This project is intended for educational and screening purposes only. It is not a medical diagnosis system.

---

## 🎯 Project Objective

The objective of SugarSense is to demonstrate how supervised machine learning can be used to build an early diabetes risk screening system.

The project covers the complete machine learning workflow:

- Data loading
- Exploratory Data Analysis
- Data preprocessing
- Feature scaling
- Train-test splitting
- Multiple model comparison
- Hyperparameter tuning
- Model evaluation
- Threshold tuning
- Feature importance analysis
- Model deployment using Streamlit
- Project overview
- Key features
- Technologies used
- ML workflow
- Model evaluation/results
- Deployment
- Limitations
- Live demo link

---

## 📊 Dataset

The dataset contains **768 patient records** and **8 clinical features**.

### Features

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | Serum insulin level |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes heredity score |
| Age | Patient age |

### Target

`Outcome`

- `0` → No diabetes
- `1` → Diabetes

---

## 🤖 Machine Learning Models

The following supervised learning algorithms were evaluated:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost

After model comparison, **Logistic Regression** was selected as the final model.

### Model Performance

**ROC-AUC: 0.823**

---

## 🎯 Threshold Tuning

Instead of relying only on the default classification threshold of `0.50`, the project evaluated multiple probability thresholds.

The selected threshold was:

```text
0.21
