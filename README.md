# Customer Churn Prediction

## Project Overview

Customer churn occurs when customers stop using a company's service. Predicting churn is important because retaining existing customers is usually less expensive than acquiring new ones.

The goal of this project is to build a machine learning model that predicts whether a customer will churn based on demographic information, service usage, and billing data.

This project uses logistic regression to classify customers as likely to churn or stay.

---

# Dataset

The dataset used is the Telco Customer Churn dataset, which contains information about telecommunications customers and whether they left the service.

Dataset characteristics:

- **Total observations:** 7043 customers  
- **Features:** 20 variables  
- **Target variable:** `Churn`

### Key Variables

| Variable | Description |
|--------|-------------|
| tenure | Number of months the customer has stayed |
| MonthlyCharges | Monthly service charges |
| TotalCharges | Total amount charged to the customer |
| Contract | Contract type |
| InternetService | Type of internet service |
| PaymentMethod | Payment method |
| Churn | Whether the customer left the service |

---

# Data Preprocessing

Several preprocessing steps were performed before training the model.

### Data Cleaning

- Converted `TotalCharges` to numeric values
- Attempt to remove missing values, there were none
- Dropped `customerID` since it does not provide predictive value or affect the model

After cleaning:

- **7032 rows**
- **20 columns**

### Encoding Categorical Variables

Categorical variables were converted into numerical values using label Encoding so they could be used in machine learning models.

### Train-Test Split

The dataset was split into:

- **80% training data**
- **20% testing data**

This produced **1407 test observations** used for evaluation.

---

# Exploratory Data Analysis

## Churn Distribution

The dataset is imbalanced, meaning more customers stayed than churned.

- **~73% stayed**
- **~27% churned**

This imbalance can bias models toward predicting that customers will stay.

---

## Tenure vs Churn

Analysis shows:

- Customers with short tenure are more likely to churn.
- Long-term customers are more likely to stay.

This suggests customer loyalty increases over time.

---

## Monthly Charges vs Churn

The analysis shows:

- Customers with higher monthly charges are more likely to churn.
- Customers with lower charges are more likely to stay.

This indicates pricing may influence churn behavior.

---

# Modeling

Two logistic regression models were built.

## Model 1: Standard Logistic Regression

Accuracy: **81.9%**

Classification Report:

| Metric | Churn = No | Churn = Yes |
|------|------|------|
| Precision | 0.85 | 0.72 |
| Recall | 0.92 | 0.54 |
| F1-score | 0.88 | 0.62 |

Interpretation:

- The model performs well at predicting customers who stay.
- However, it struggles to identify customers who churn.
- Recall for churn is only 54%, meaning many churn cases are missed.

---

## Model 2: Balanced Logistic Regression

To address class imbalance, the model was trained using:

```python
class_weight = "balanced"
```

Accuracy: **77.4%**

Classification Report:

| Metric | Churn = No | Churn = Yes |
|------|------|------|
| Precision | 0.91 | 0.56 |
| Recall | 0.77 | 0.80 |
| F1-score | 0.83 | 0.66 |

Interpretation:

- Overall accuracy decreased slightly compared to the standard logistic regression model.
- However, the model significantly improved churn detection.
- Recall for churn increased from 54% to 80%, meaning the model correctly identifies most customers who will leave the service.

This tradeoff is often acceptable in business applications because usually missing a churned customer can be more costly than incorrectly predicting churn.

---

# Model Evaluation

## Confusion Matrix

Confusion matrices were used to evaluate prediction results.

| | Predicted Stay | Predicted Churn |
|---|---|---|
| Actual Stay | True Negative | False Positive |
| Actual Churn | False Negative | True Positive |

Reducing false negatives is particularly important because they represent customers who churn but were predicted to stay.

The balanced logistic regression model reduced false negatives compared to the original model, making it more effective for identifying customers at risk of leaving.

---

# ROC Curve and AUC

The balanced model achieved:

AUC Score = **0.85**

AUC (Area Under the Curve) measures how well the model distinguishes between churn and non-churn customers.

Interpretation:

- **0.5** → Random guessing  
- **0.7 – 0.8** → Good model  
- **0.8 – 0.9** → Very good model  

An AUC score of **0.85** indicates the model has strong predictive performance.

---

# Business Insights

Key insights from the analysis include:

- Customers with short tenure are more likely to churn
- Customers with higher monthly charges show higher churn probability
- Dataset imbalance can reduce model performance
- Using class balancing improves churn detection

Potential strategies to reduce churn:

- Offer discounts for customers with high monthly charges
- Provide loyalty incentives for newer customers
- Identify high-risk customers early for targeted retention campaigns

---

# Conclusion

This project developed a logistic regression model to predict customer churn using telecommunications customer data.

The balanced logistic regression model performed better at detecting churn, achieving:

| Metric | Value |
|------|------|
| Accuracy | 77% |
| Recall (Churn Detection) | 80% |
| AUC Score | 0.85 |

Although the standard model had slightly higher accuracy, the balanced model is more valuable because it identifies more customers who are likely to leave, allowing businesses to intervene earlier.

---

# Future Improvements

Future improvements could include testing more advanced machine learning models such as:

- Random Forest
- Gradient Boosting
- XGBoost

These models may capture more complex patterns and further improve prediction performance.

---

# Tools Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn
