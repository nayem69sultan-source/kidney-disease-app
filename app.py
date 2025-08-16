import streamlit as st
import numpy as np
from tensorflow import keras

st.set_page_config(page_title="Kidney Disease Prediction", layout="centered")
st.title("Kidney Disease Prediction Web App")

# -------------------------
# Load Keras model
# -------------------------
keras_model = keras.models.load_model('my_model.keras')
st.success("Keras model loaded successfully!")

# -------------------------
# Input features
# -------------------------
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

# -------------------------
# Form for inputs
# -------------------------
with st.form(key='input_form'):
    user_input = []
    for feature in feature_names:
        min_val, max_val, default_val = feature_ranges[feature]
        if feature == "age":
            value = st.slider(f"{feature}", min_value=int(min_val), max_value=int(max_val), value=int(default_val))
        else:
            value = st.slider(f"{feature}", min_value=float(min_val), max_value=float(max_val), value=float(default_val))
        user_input.append(value)

    submitted = st.form_submit_button("Predict CKD")

# -------------------------
# Prediction
# -------------------------
if submitted:
    input_array = np.array([user_input])
    
    # Predict directly on raw values
    prediction = keras_model.predict(input_array)
    prob = float(prediction[0][0])
    
    st.write("Prediction probability (CKD):", round(prob, 4))
    
    # Determine risk level
    if prob < 0.25:
        risk = "Low"
        color = "green"
    elif prob < 0.5:
        risk = "Medium"
        color = "yellow"
    elif prob < 0.75:
        risk = "High"
        color = "orange"
    else:
        risk = "Very High"
        color = "red"
    
    st.success(f"Predicted class: {'CKD' if prob >= 0.5 else 'Not CKD'}")
    st.warning(f"Risk Level: {risk}")
    
    # Colored progress bar
    st.markdown(f"""
        <div style="background-color:#ddd; border-radius:5px; padding:3px;">
            <div style="width:{prob*100}%; background-color:{color}; text-align:center; padding:5px 0; border-radius:5px; color:white;">
                {round(prob*100, 1)}%
            </div>
        </div>
    """, unsafe_allow_html=True)
