# app.py
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import joblib

st.set_page_config(page_title="Kidney Disease Prediction", layout="centered")
st.title("Kidney Disease Prediction Web App")

# -------------------------
# Load Keras model
# -------------------------
try:
    keras_model = keras.models.load_model('my_model.keras')
    st.success("Keras model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# -------------------------
# Load saved scaler
# -------------------------
try:
    scaler = joblib.load("scaler.save")
except Exception as e:
    st.warning("Scaler not found. Using default StandardScaler (not recommended).")
    scaler = StandardScaler()

# -------------------------
# Input section
# -------------------------
st.subheader("Enter patient details")

feature_names = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot']

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

cols = st.columns(2)
user_input = []

for idx, feature in enumerate(feature_names):
    min_val, max_val, default_val = feature_ranges[feature]
    col = cols[idx % 2]
    
    if feature in ['age', 'bp', 'al', 'su', 'bgr', 'bu', 'sod']:
        value = col.number_input(f"{feature}", min_value=int(min_val), max_value=int(max_val), value=int(default_val))
    else:
        value = col.number_input(f"{feature}", min_value=float(min_val), max_value=float(max_val), value=float(default_val))
    
    user_input.append(value)

input_array = np.array([user_input])
input_scaled = scaler.transform(input_array)  # only transform

# -------------------------
# Predict button
# -------------------------
if st.button("Predict CKD Risk"):
    prediction = keras_model.predict(input_scaled)
    prob = float(prediction[0][0])
    
    # Determine risk level
    if prob < 0.33:
        risk = "Low"
        color = "green"
    elif prob < 0.66:
        risk = "Medium"
        color = "orange"
    else:
        risk = "High"
        color = "red"
    
    st.write(f"**CKD Probability:** {round(prob, 4)}")
    st.markdown(f"**Predicted Risk Level:** <span style='color:{color}; font-size:20px'>{risk}</span>", unsafe_allow_html=True)
    
    # Visual progress bar
    st.progress(prob)
