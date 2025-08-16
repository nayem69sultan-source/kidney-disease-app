# app.py
import streamlit as st
import numpy as np
from tensorflow import keras
import joblib

st.set_page_config(page_title="Kidney Disease Prediction", layout="centered")
st.title("Kidney Disease Prediction Web App")

# -------------------------
# Load Keras model
# -------------------------
keras_model = keras.models.load_model('my_model.keras')
st.success("Keras model loaded successfully!")

# -------------------------
# Load saved scaler
# -------------------------
scaler = joblib.load('scaler.save')

# -------------------------
# Feature definitions
# -------------------------
feature_names = [
    'age', 'bp', 'sg', 'al', 'su', 
    'bgr', 'bu', 'sc', 'sod', 'pot'
]

feature_ranges = {
    'age': (1, 100, 50),
    'bp': (50, 200, 80),
    'sg': (1.005, 1.030, 1.015),
    'al': (0, 5, 0),
    'su': (0, 5, 0),
    'bgr': (50, 500, 100),
    'bu': (1, 200, 15),
    'sc': (0.1, 20, 1.0),
    'sod': (100, 160, 140),
    'pot': (2.5, 8, 4.5)
}

# -------------------------
# Streamlit form
# -------------------------
with st.form("patient_form"):
    st.subheader("Enter patient details")
    user_input = []
    
    for feature in feature_names:
        min_val, max_val, default_val = feature_ranges[feature]
        if feature in ['age', 'bp', 'al', 'su', 'bgr', 'bu', 'sod']:
            value = st.number_input(f"{feature}", min_value=int(min_val), max_value=int(max_val), value=int(default_val))
        else:
            value = st.number_input(f"{feature}", min_value=float(min_val), max_value=float(max_val), value=float(default_val), format="%.3f")
        user_input.append(value)
    
    submitted = st.form_submit_button("Predict CKD Risk")

# -------------------------
# Prediction
# -------------------------
if submitted:
    input_array = np.array([user_input])
    input_scaled = scaler.transform(input_array)

    prediction = keras_model.predict(input_scaled)
    
    # Multiclass handling
    risk_levels = ["Low Risk", "Medium Risk", "High Risk"]
    result_index = np.argmax(prediction)
    result = risk_levels[result_index]
    
    st.success(f"Predicted CKD Risk Level: {result}")
    st.subheader("Prediction probabilities:")
    for level, prob in zip(risk_levels, prediction[0]):
        st.write(f"{level}: {prob:.4f}")
