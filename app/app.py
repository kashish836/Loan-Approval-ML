import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('../models/loan_model.pkl', 'rb'))
scaler = pickle.load(open('../models/scaler.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Loan Predictor", page_icon="💰", layout="centered")

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💰 Loan Approval Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Check if your loan will be approved instantly 🚀</p>", unsafe_allow_html=True)

st.write("---")

# Inputs in columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    married = st.selectbox("💍 Married", ["Yes", "No"])
    dependents = st.number_input("👶 Dependents", 0, 5)
    education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])

with col2:
    self_employed = st.selectbox("💼 Self Employed", ["Yes", "No"])
    app_income = st.number_input("💰 Applicant Income")
    coapp_income = st.number_input("💵 Coapplicant Income")
    loan_amount = st.number_input("🏦 Loan Amount")

loan_term = st.number_input("⏳ Loan Term")
credit_history = st.selectbox("📊 Credit History", ["Good", "Bad"])
property_area = st.selectbox("📍 Property Area", ["Urban", "Semiurban", "Rural"])

st.write("---")

# Convert inputs to numerical
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good" else 0

# Property Area encoding
if property_area == "Urban":
    property_area = 2
elif property_area == "Semiurban":
    property_area = 1
else:
    property_area = 0

if st.button("🚀 Predict Loan Status"):
    
    st.write("Button clicked ✅")

    input_data = np.array([[gender, married, dependents, education, self_employed,
                            app_income, coapp_income, loan_amount, loan_term,
                            credit_history, property_area]])

    st.write("Input Data:", input_data)

    input_scaled = scaler.transform(input_data)
    st.write("Scaled Data:", input_scaled)

    prediction = model.predict(input_scaled)
    st.write("Prediction Raw:", prediction)

    if prediction[0] == 1:
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Not Approved")