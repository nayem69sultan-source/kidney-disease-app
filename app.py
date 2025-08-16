# app.py
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

st.set_page_config(page_title="Kidney Disease Prediction", layout="centered")
st.title("Kidney Disease Prediction Web App")

# -------------------------
# Load Keras model
# -------------------------
keras_model = keras.models.load_model('my_model.keras')
st.success("Keras model loaded successfully!")

# -------------------------
# Manual input for prediction
# -------------------------
st.subheader("Enter patient details to predict CKD")

# Define features your Keras model expects
feature_names = [
    'age', 'bp', 'sg', 'al', 'su', 
    'bgr', 'bu', 'sc', 'sod', 'pot'
]

# Define min, max, and default for sliders
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

user_input = []
for feature in feature_names:
    min_val, max_val, default_val = feature_ranges[feature]
    value = st.slider(f"{feature}", min_value=float(min_val), max_value=float(max_val), value=float(default_val))
    user_input.append(value)

input_array = np.array([user_input])

# -------------------------
# Optional: scale inputs (use same scaler as training)
# -------------------------
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_array)  # Replace with saved scaler if available

# -------------------------
# Predict button
# -------------------------
if st.button("Predict CKD"):
    prediction = keras_model.predict(input_scaled)
    prob = float(prediction[0][0])
    
    # Display probability as progress bar
    st.subheader("Prediction Probability")
    st.progress(prob)
    
    # Color-coded risk level
    if prob < 0.3:
        risk = "Low Risk"
        color = "green"
    elif prob < 0.7:
        risk = "Medium Risk"
        color = "orange"
    else:
        risk = "High Risk"
        color = "red"
    
    st.markdown(f"<h3 style='color:{color};'>Risk Level: {risk}</h3>", unsafe_allow_html=True)
    
    # Predicted class
    result = "CKD" if prob >= 0.5 else "Not CKD"
    st.success(f"Predicted class: {result}")
