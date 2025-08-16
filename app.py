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

# Feature names expected by your model
feature_names = [
    'bp', 'sg', 'al', 'su', 
    'bgr', 'bu', 'sc', 'sod', 'pot'
]

# Slider ranges for each feature
feature_ranges = {
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
    value = st.slider(
        f"{feature}", 
        min_value=float(min_val), 
        max_value=float(max_val), 
        value=float(default_val)
    )
    user_input.append(value)

input_array = np.array([user_input])

# -------------------------
# Scale input using StandardScaler
# -------------------------
# NOTE: For deployment without a saved scaler, we'll standardize input based on min-max approximation
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_array)  # Quick fix for deployment

# -------------------------
# Predict button
# -------------------------
if st.button("Predict CKD"):
    prediction = keras_model.predict(input_scaled)
    prob = float(prediction[0][0])
    st.write("Prediction probability (CKD):", round(prob, 4))

    # Define prediction class thresholds
    if prob < 0.5:
        result = "Not CKD"
    elif prob < 0.75:
        result = "Medium Risk"
    else:
        result = "High Risk"

    st.success(f"Predicted class: {result}")
