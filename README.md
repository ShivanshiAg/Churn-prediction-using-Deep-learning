# Churn-prediction-using-Deep-learning(ANN Classification)
🤖 ANN-Based Salary & Churn Prediction

Welcome to the **Artificial Neural Network (ANN)** based dual-project repository!  
This repo includes two complete pipelines:

- 🧾 **Customer Churn Prediction** (Classification)
- 💼 **Salary Prediction** (Regression)

Both use deep learning techniques with structured data, and include preprocessing, model training, evaluation, and saved artifacts for deployment or reuse.

---

## 🧭 Project Flow Overview

### 🔄 Common Pipeline Steps

1. **Data Ingestion** – Load CSV data
2. **Preprocessing** – Handle missing values, encode categorical features, scale numerical variables
3. **Model Building** – Construct ANN using Keras
4. **Model Training** – Fit model on training data
5. **Model Evaluation** – Use appropriate metrics
6. **Model Saving** – Save models and scalers using `.h5` and `.pkl` files
7. **Inference** – Use saved model for predictions
8. **(Optional)** Deployment using Streamlit (`app.py`)

---

## 📊 Project 1: Customer Churn Prediction

> Predict whether a bank customer will churn based on demographics and financial attributes.

### 🧾 Features Used
- Gender
- Geography
- Age
- Tenure
- Credit Score
- Balance
- Number of Products
- Active Member status
- Estimated Salary

### 📈 Process Flow

**Step 1**: Load dataset (`Churn_Modelling.csv`)  
**Step 2**: Encode categorical features (e.g., gender, geography)  
**Step 3**: Scale features using `StandardScaler`  
**Step 4**: Build ANN with:
- Input layer (number of features)
- Hidden layers with ReLU activation
- Output layer with Sigmoid activation

**Step 5**: Evaluate model with:
- Accuracy
- Confusion Matrix
- Precision, Recall, F1 Score
- ROC-AUC

**Step 6**: Save model (`model.h5`), encoders, and scalers  
**Step 7**: Run inference using `prediction.ipynb`

---

## 📊 Project 2: Salary Prediction

> Predict the salary of individuals based on structured input features.

### 🧾 Features Used
- Age
- Years of Experience
- Education (if present)
- Industry / Region (optional)

### 📈 Process Flow

**Step 1**: Load dataset  
**Step 2**: Handle missing values (if any)  
**Step 3**: Scale features using `StandardScaler`  
**Step 4**: Build ANN with:
- Input layer (features)
- Hidden layers with ReLU
- Output layer with linear activation (for regression)

**Step 5**: Evaluate using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

**Step 6**: Save model as `regression_model.h5`  
**Step 7**: Run predictions using `prediction.ipynb`

---

## 🗂️ Repository Structure

├── app.py # Optional Streamlit / deployment script

├── Churn_Modelling.csv # Churn dataset

├── salary_regression.ipynb # Salary ANN notebook

├── data_tranformation+model_building.ipynb # Churn ANN pipeline

├── hyperparametertuning_ANN.ipynb # Tuning experiment

├── prediction.ipynb # Inference notebook

├── model.h5 # Churn ANN model

├── regression_model.h5 # Salary ANN model

├── scaler.pkl # Feature scaler

├── encode_geography.pkl # OneHot encoder

├── label_encode_gender.pkl # Label encoder

├── requirements.txt # Dependencies

├── runtime.txt # Python version (for Streamlit cloud)

├── logs/, regressionlogs/ # Training logs

