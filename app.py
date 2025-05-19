import streamlit as st
import joblib
import numpy as np

# Load your saved model
model = joblib.load('exam_predictor_model.pkl')

# Title of your app
st.title("Exam Predictor App")

# Create input fields for the features your model needs
# Example: If your model needs 2 features: 'feature1' and 'feature2'

assessment1 = st.number_input('Assessment 1 Marks:', min_value=0.0, max_value=100.0, value=50.0)
assessment2 = st.number_input('Assessment 2 Marks:', min_value=0.0, max_value=100.0, value=50.0)
assessment3 = st.number_input('Assessment 3 Marks:', min_value=0.0, max_value=100.0, value=50.0)

# Prediction
if st.button('Predict Exam Marks'):
    input_data = np.array([[assessment1, assessment2, assessment3]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Final Exam Marks: {prediction:.2f}")