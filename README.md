# Customer Churn Prediction App 📈

This project is an interactive web application built with Streamlit that predicts customer churn. The app uses a `RandomForestClassifier` model trained on a sample telecommunications dataset to determine the likelihood of a customer leaving the service.

-----

## 🚀 Live Demo

https://churn-prediction-app-6cfm47xmeiclkljmpkw6ri.streamlit.app/

-----

## ✨ Features

  * **Interactive Prediction**: Use the sidebar sliders to input customer data (tenure, monthly charges, total charges) and get a real-time churn prediction.
  * **Data Visualization**: Includes interactive charts from Plotly to explore the distribution of churn and the relationship between customer tenure and monthly charges.
  * **Machine Learning Integration**: The app loads a pre-trained Scikit-learn model to make its predictions.

-----

## 🛠️ How to Run This Project Locally

Follow these steps to set up and run the project on your own machine.

### 1\. Clone the Repository

```bash
git clone https://github.com/[YOUR-USERNAME]/[YOUR-REPOSITORY-NAME].git
cd [YOUR-REPOSITORY-NAME]
```

### 2\. Install Dependencies

It's recommended to use a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install the required libraries
pip install -r requirements.txt
```

### 3\. Run the Streamlit App

Once the dependencies are installed, run the dashboard script:

```bash
streamlit run dashboard.py
```

The application will open in a new tab in your web browser.

-----

## 📂 Files in This Repository

  * **`dashboard.py`**: The main script for the Streamlit web application.
  * **`train_churn_model.py`**: A script to train the Random Forest model and save it.
  * **`churn_model.pkl`**: The pre-trained and saved machine learning model file.
  * **`telco_churn.csv`**: The sample dataset used for training the model.
  * **`requirements.txt`**: A list of the necessary Python libraries for the project.
  * **`README.md`**: This file, providing an overview and instructions for the project.
