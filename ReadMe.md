# 📊 RavenStack Churn Prediction: An Optimized Machine Learning Approach

## 1\. Project Overview

This project focuses on building a highly sensitive predictive model to identify high-risk customer accounts (churn probability $\\ge 0.25$) 90 days in advance for RavenStack, a B2B SaaS company.

The primary goal was to maximize **Recall** to minimize customer attrition.

## 2\. Methodology \& Key Technologies

The entire pipeline was built using Python, focusing on robustness and production readiness.

* **Data Sources:** Accounts, Subscriptions, Feature Usage, Support Tickets, and Churn Events (5 tables).
* **Feature Engineering:** $\\mathbf{35}$ engineered features were created, including rolling averages (30d/90d usage), subscription velocity, and support burden metrics.
* **Modeling:** **Gradient Boosting Classifier (GBC)**, embedded in an $\\mathbf{Imbalanced-learn : Pipeline}$.
* **Imbalance Handling:** **SMOTE** (Synthetic Minority Over-sampling Technique) was applied to the training data to improve the model's sensitivity to the minority churn class.
* **Final Output:** A trained, serialized model pipeline: `ravenstack\_churn\_predictor.joblib`.

## 3\. Key Results \& Business Impact

The final model's strategy was optimized using a **$0.25$ probability threshold** to prioritize retention efforts.

| Metric | Business Goal | Final Result | Interpretation |
| :--- | :--- | :--- | :--- |
| \*\*Churn Recall\*\* | MAXIMIZED (Minimize lost customers) | $\\mathbf{65.91\\%}$ | The model correctly identifies \*\*2 out of every 3\*\* actual churners. |
| \*\*Prediction Threshold\*\* | Optimized Strategy | $\\mathbf{0.25}$ | Accounts $\\ge 25\\%$ churn probability are flagged for $\\mathbf{immediate \\: retention \\: action}$. |

### Top 3 Churn Drivers (Feature Importance)

1. **tenure\_days:** The most critical factor; accounts are highly susceptible to early churn.
2. **plan\_tier\_Basic:** Basic plan users show higher instability and lower perceived value.
3. **billing\_frequency\_monthly:** Low commitment is a significant risk indicator.

## 4\. How to Use the Model (Production Scoring)

The final model is saved as a single object, ensuring all preprocessing steps are included.

### Requirements

```bash
pip install pandas scikit-learn imbalanced-learn joblib

