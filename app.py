import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered",
)

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Sidebar
with st.sidebar:
    st.image("https://i.imgur.com/9l9q1Zm.png", width=200)  # Optional: Company logo
    st.markdown("### 🧠 AI Model Info")
    st.markdown("""
    - Model: ANN (Keras)
    - Task: Predict customer churn
    - Input features: Demographics, account info
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# Main title
st.title("📉 Customer Churn Prediction")
st.caption("Use this tool to predict whether a customer is likely to churn based on input details.")

# --- Form UI ---
with st.form("input_form"):
    st.subheader("📋 Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92, 35)
        credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=1000, value=600)
        tenure = st.slider('📆 Tenure (years)', 0, 10, 3)

    with col2:
        balance = st.number_input('🏦 Balance', min_value=0.0, format="%.2f")
        estimated_salary = st.number_input('💰 Estimated Salary', min_value=0.0, format="%.2f")
        num_of_products = st.slider('📦 Number of Products', 1, 4, 1)
        has_cr_card = st.selectbox('💳 Has Credit Card?', ['No', 'Yes'])
        is_active_member = st.selectbox('✅ Is Active Member?', ['No', 'Yes'])

    submitted = st.form_submit_button("Predict")

# --- Prediction Logic ---
if submitted:
    # Map categorical values
    has_cr_card_bin = 1 if has_cr_card == 'Yes' else 0
    is_active_bin = 1 if is_active_member == 'Yes' else 0

    # Create input dataframe
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card_bin],
        'IsActiveMember': [is_active_bin],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0][0]

    # Display result
    # Display result
st.subheader("🧾 Prediction Result")
st.progress(int(prediction * 100))  # progress bar from 0 to 100%

if prediction > 0.5:
    st.error(f"⚠️ The customer is likely to churn. (Probability: {prediction:.2f})")
else:
    st.success(f"✅ The customer is not likely to churn. (Probability: {prediction:.2f})")

