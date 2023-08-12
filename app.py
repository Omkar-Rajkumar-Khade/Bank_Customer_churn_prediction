import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the trained model
model = load_model('models/churn_prediction_model.h5')

# Load the scaler
scaler = StandardScaler()

# Load the original dataset
original_data = pd.read_csv('dataset\Bank Customer Churn Prediction.csv')
original_data.drop(columns=['customer_id'], inplace=True)

# Get the feature columns you want to use for scaling
feature_columns = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']

# Extract and scale the feature values from the original data
scaled_data = scaler.fit_transform(original_data[feature_columns])

# Streamlit UI
st.title("Bank Customer Churn Prediction")

# Sidebar input
credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
age = st.sidebar.slider("Age", 18, 100, 30)
tenure = st.sidebar.slider("Tenure", 0, 10, 5)
balance = st.sidebar.slider("Balance", 0, 250000, 50000)
products_number = st.sidebar.slider("Number of Products", 1, 4, 1)
credit_card = st.sidebar.checkbox("Has Credit Card")
active_member = st.sidebar.checkbox("Active Member")
estimated_salary = st.sidebar.slider("Estimated Salary", 0, 200000, 80000)

# Convert "Has Credit Card" and "Active Member" to 0 or 1
credit_card = 1 if credit_card else 0
active_member = 1 if active_member else 0

# Scale the input data based on original dataset distribution
input_data = np.array([credit_score, age, tenure, balance, products_number, credit_card, active_member, estimated_salary])
input_data_scaled = scaler.transform(input_data.reshape(1, -1))

# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]
    churn_result = "Churn" if churn_probability > 0.5 else "Not Churn"
    
    st.write(f"Churn Probability: {churn_probability:.2f}")
    st.write(f"Prediction: {churn_result}")
