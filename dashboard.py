# Step 3: Build the interactive Dashboard 📊
# We'll use Streamlit to create a simple dashboard that allows users to input data and see predictions.
# Streamlit is a powerful library for building web applications in Python, especially for data science projects
import streamlit as st
import pandas as pd 
import numpy as np
import joblib  # Used to load the trained model
import plotly.express as px  # Used for creating interactive plots

# ---1. Load the trained model---
# Load the trained model from the file we saved earlier
model = joblib.load('churn_model.pkl')
df = pd.read_csv('telco_churn.csv')

# ---2. Set up the Streamlit app---
st.title("Customer Churn Prediction Dashboard")  # Set the title of the dashboard
st.markdown("This dashboard predicts whether a customer will churn based on their tenure, monthly charges, and total charges.") 

# ---3. Data Visualization Section---
st.header('Exploratory Data Analysis (EDA)')  # Header for the EDA section

# Churn Distrribution Pie Chart
churn_dist = df['Churn'].value_counts()
fig_pie = px.pie(values=churn_dist, names=['No Churn', 'Churn'], title='Churn Distribution')
st.plotly_chart(fig_pie, use_container_width=True)  # Display the pie chart

# Scatter Plot of Tenure vs Monthly Charges
st.subheader('Tenure vs Monthly Charges')  # Subheader for the scatter plot
fig_scatter = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn', title='Tenure vs Monthly Charges by Churn Status')
st.plotly_chart(fig_scatter, use_container_width=True)  # Display the scatter plot

# ---4. Prediction Section in Sidebar---
st.sidebar.header('🔮 Make a Prediction')
st.sidebar.markdown('Enter customer details to predict churn.')

# Input fields for user to enter customer data
tenure = st.sidebar.slider('Tenure (months)', 1, 72, 24)  # Slider for tenure
monthly_charges = st.sidebar.slider('Monthly Charges ($)', 18, 120,70)  # Slider for monthly charges
total_charges = st.sidebar.slider('Total Charges ($)', 18, 8700, 1400)  # Slider for total charges

# Button to make a prediction
if st.sidebar.button('Predict Churn'):
    # Prepare the input for the model
    input_data = np.array([[tenure, monthly_charges, total_charges]])  # Create a 2D array for the model input

    # Make the prediction using the loaded model
    prediction = model.predict(input_data)  # Use the model to predict churn
    prediction_proba = model.predict_proba(input_data)  # Get the probability of churn

    # Display the prediction result
    if prediction[0] == 1:
        st.error("The customer is likely to churn! 🚨")  # If the prediction is 1 (churn)
    else:
        st.success("The customer is likely to stay! 😊")

st.write(f"**Confidence:** {prediction_proba[0][prediction[1]]*100:.2f}% for churn, {prediction_proba[0][0]*100:.2f}% for no churn")  # Display the confidence levels
st.write('Prediction Probabilities:', prediction_proba)  # Show the prediction probabilities

