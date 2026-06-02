import streamlit as st
import pickle
import numpy as np

# Set up the web page styling
st.set_page_config(page_title="Loan Approval Prediction System", page_icon="💰", layout="centered")

# 1. Load the trained model and label encoder from your folder
@st.cache_resource
def load_assets():
    with open('loan_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('label_encoder.pkl', 'rb') as encoder_file:
        le = pickle.load(encoder_file)
    return model, le

try:
    model, le = load_assets()
except FileNotFoundError:
    st.error("⚠️ Missing Model Files! Please ensure 'loan_model.pkl' and 'label_encoder.pkl' are in the exact same folder as this app.py file.")
    st.stop()

st.title("💰 Loan Approval Prediction App")
st.write("This application uses your trained Random Forest model to predict if a loan application will be approved or rejected.")
st.markdown("---")

st.subheader("📋 Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed?", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=5000, step=100)
    coapplicant_income = st.number_input("Co-applicant Monthly Income ($)", min_value=0, value=0, step=100)
    loan_amount = st.number_input("Loan Amount ($ in Thousands, e.g. 120 = $120,000)", min_value=0, value=120, step=5)
    loan_term = st.selectbox("Loan Amount Term (In Days)", [360, 180, 120, 84, 60])
    credit_history = st.selectbox("Credit History Score", ["Good (1.0)", "Bad (0.0)"])

# Property Area input
property_area = st.selectbox("Property Location Area", ["Urban", "Semiurban", "Rural"])

# 2. When the user clicks the button, process inputs and predict
if st.button("Analyze Loan Application", type="primary"):
    
    # Use the loaded label encoder to change text options into the exact numbers the model expects
    gender_encoded = le.transform([gender])[0]
    married_encoded = le.transform([married])[0]
    education_encoded = le.transform([education])[0]
    self_employed_encoded = le.transform([self_employed])[0]
    property_encoded = le.transform([property_area])[0]
    
    # Safely convert dependents string to a number
    if dependents == "3+":
        dependents_encoded = 3
    else:
        dependents_encoded = int(dependents)
        
    credit_history_encoded = 1.0 if "Good" in credit_history else 0.0

    # Arrange features into the EXACT order your Colab model expects
    features = np.array([[
        gender_encoded,
        married_encoded,
        dependents_encoded,
        education_encoded,
        self_employed_encoded,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history_encoded,
        property_encoded
    ]])

    # 3. Generate Prediction
    prediction = model.predict(features)
    
    st.markdown("---")
    # Display the final outcome
    if prediction[0] == 1 or prediction[0] == 'Y':
        st.success("🎉 **Status: Loan Approved!** The applicant meets the safe parameters for credit confirmation.")
        st.balloons()
    else:
        st.error("❌ **Status: Loan Rejected.** The applicant's profile has high risk metrics.")
