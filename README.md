# 🏥 Patient No-Show Predictive Model

## 📌 Overview
This project features a predictive classification pipeline designed to identify high-risk healthcare appointments before they happen. By accurately forecasting patient no-shows, clinic staff can proactively reach out to at-risk patients. The model optimizes for business value by balancing the low cost of a preventative phone call against the high financial cost of a lost clinical hour.

## ⚙️ Data & Feature Engineering
The model processes historical appointment data (`historical_appointments.csv`) to surface behavioral patterns. Core features include:
* **Lead Time (Engineered):** The calculated number of days between the booking date and the actual appointment. *(Note: This emerged as the strongest predictor of attendance).*
* **Age:** Patient demographic data.
* **SMS Received:** A binary indicator of whether a text reminder was sent and received.
* **Clinic Location:** Ordinal encoded categorical data representing the specific facility.

## 🧠 Modeling Approach
* **Algorithm:** Random Forest Classifier built with `scikit-learn`.
* **Custom Classification Thresholds:** Instead of relying on standard default thresholds, the model's decision boundary is deliberately shifted. This customization allows the system to maximize the identification of actual no-shows (Recall) while maintaining an acceptable rate of false alarms (Precision).

<img width="615" height="461" alt="Screenshot 2026-04-25 at 2 04 04 PM" src="https://github.com/user-attachments/assets/c7a729b6-f546-48a1-9790-272d288dd5a5" />

## 📊 Key Results
The model successfully identifies at-risk appointments with the following performance metrics:
* **Precision:** [19]% 
* **Recall:** [82]% 
* **Primary Driver:** Lead Time


## 🚀 How to Run

1. Clone the repository and ensure your environment has the required dependencies installed:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   

2. Ensure the historical_appointments.csv dataset is located in the same root directory as the script.
   
   
3. Execute the pipeline:
   ```python train_model.py
