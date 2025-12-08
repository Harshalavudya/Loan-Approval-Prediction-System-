import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏦 Loan Application Success Prediction (Random Forest)")

st.subheader("Enter Applicant Details")

# Collect user inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
co_applicant_income = st.number_input("Co-Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_terms = st.selectbox("Loan Term (months)", [180, 360])
credit_history = st.selectbox("Credit History", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

if st.button("Predict"):
    # Map categorical inputs to numeric
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = 1 if credit_history == "Yes" else 0
    property_area = 0 if property_area == "Rural" else 1 if property_area == "Semiurban" else 2

    # Prepare input array
    user_inputs = [
        gender, married, dependents, education, self_employed,
        applicant_income, co_applicant_income, loan_amount,
        loan_terms, credit_history, property_area
    ]
    
    user_inputs = np.array(user_inputs).reshape(1, -1)
    user_inputs_scaled = scaler.transform(user_inputs)

    # Prediction
    prediction = model.predict(user_inputs_scaled)
    prediction_prob = model.predict_proba(user_inputs_scaled)[0][1]

    eligibility = "✅ Eligible" if prediction == 1 else "❌ Not Eligible"

    st.subheader("Prediction Result")
    st.write(f"**Loan Status:** {eligibility}")
    st.write(f"**Approval Probability:** {prediction_prob:.2%}")
