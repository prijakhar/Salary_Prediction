import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💰",
    layout="centered"
)

st.sidebar.title("About")

st.sidebar.info(
    """
    This application predicts an employee's salary
    based on:

    • Experience Level

    • Employment Type

    • Job Title

    • Company Size

    • Remote Ratio

    Model Used:
    Linear Regression
    """
)

model = joblib.load("models/salary_prediction_model.pkl")

scaler = joblib.load("models/scaler.pkl")

model_columns = joblib.load("models/model_columns.pkl")

st.title("💰 Employee Salary Prediction")
 
st.markdown(
"""
Predict an employee's salary using a trained Machine Learning model.

Fill in the employee details below and click **Predict Salary**.
"""
)

st.divider()

employment_type = st.selectbox(
    "Employment Type",
    ["FT", "PT", "CT", "FL"]
)

job_title = st.selectbox(
    "Job Title",
    [
        "Data Scientist",
        "Data Analyst",
        "Machine Learning Engineer",
        "AI Engineer",
        "Data Engineer"
    ]
)

company_size = st.selectbox(
    "Company Size",
    ["S", "M", "L"]
)

remote_ratio = st.slider(
    "Remote Ratio",
    0,
    100,
    100
)
predict_button = st.button("Predict Salary")

if predict_button:
    st.subheader("Employee Details")

    # st.write(input_data)

    # Create DataFrame from user input
    input_data = pd.DataFrame({
        # "experience_level": [experience_level],
        "employment_type": [employment_type],
        "job_title": [job_title],
        "company_size": [company_size],
        "remote_ratio": [remote_ratio]
    })

    # One-Hot Encoding
    input_encoded = pd.get_dummies(input_data)

    # Match training columns
    input_encoded = input_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Scale input
    input_scaled = scaler.transform(input_encoded)

    # Predict salary
    prediction = model.predict(input_scaled)

    # Show result
    st.success(
    f"💰 Predicted Annual Salary: ${prediction[0]:,.2f}"
)
    
    st.divider()

st.caption(
    "Developed by PREETI JAKHAR | Machine Learning Project using Python & Streamlit"
)