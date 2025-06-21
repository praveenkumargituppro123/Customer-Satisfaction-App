import subprocess
import sys
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install_packages():
    packages = [
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn'
    ]

    for package in packages:
        try:
            __import__(package)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"Installing {package}...")
            install_package(package)

class CustomerSatisfactionPredictor:
    def __init__(self, df):
        # Prepare features and target
        X = df[['Product Purchased Encoded', 'Ticket Type Encoded',
                'Ticket Priority Encoded', 'Ticket Status Encoded', 'Customer Age']]
        y = df['Customer Satisfaction Rating']

        # Scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create and train the model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # Train the model
        self.model.fit(X_scaled, y)

        # Create feature mappings
        self.feature_mappings = {
            'products': df['Product Purchased'].unique().tolist(),
            'ticket_types': df['Ticket Type'].unique().tolist(),
            'priorities': df['Ticket Priority'].unique().tolist(),
            'statuses': df['Ticket Status'].unique().tolist()
        }

        # Create reverse mappings for encoding
        self.product_to_code = {prod: idx for idx, prod in enumerate(self.feature_mappings['products'])}
        self.ticket_type_to_code = {tt: idx for idx, tt in enumerate(self.feature_mappings['ticket_types'])}
        self.priority_to_code = {pri: idx for idx, pri in enumerate(self.feature_mappings['priorities'])}
        self.status_to_code = {stat: idx for idx, stat in enumerate(self.feature_mappings['statuses'])}

    def predict(self, product, ticket_type, priority, status, age):
        # Encode features
        product_encoded = self.product_to_code.get(product)
        ticket_type_encoded = self.ticket_type_to_code.get(ticket_type)
        priority_encoded = self.priority_to_code.get(priority)
        status_encoded = self.status_to_code.get(status)

        if any(x is None for x in [product_encoded, ticket_type_encoded, priority_encoded, status_encoded]):
            raise ValueError("Invalid input values provided")

        # Create feature array
        features = np.array([
            product_encoded,
            ticket_type_encoded,
            priority_encoded,
            status_encoded,
            age
        ]).reshape(1, -1)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Make prediction
        prediction = self.model.predict(features_scaled)[0]

        return {
            'prediction': float(prediction),
            'satisfaction_level': self._get_satisfaction_level(prediction),
            'details': {
                'product': product,
                'ticket_type': ticket_type,
                'priority': priority,
                'status': status,
                'customer_age': age
            }
        }

    def _get_satisfaction_level(self, prediction):
        if prediction >= 4:
            return 'Very Satisfied'
        elif prediction >= 3:
            return 'Satisfied'
        elif prediction >= 2:
            return 'Neutral'
        elif prediction >= 1:
            return 'Dissatisfied'
        else:
            return 'Very Dissatisfied'

    def get_valid_values(self):
        return self.feature_mappings

def login_page():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state['logged_in'] = True
        else:
            st.error("Invalid username or password")

def main_app():
    st.set_page_config(page_title="Customer Satisfaction Predictor", layout="wide")

    st.title('Customer Satisfaction Predictor')
    st.markdown("""
    <style>
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        predictor = CustomerSatisfactionPredictor(df)

        # Sidebar for user inputs
        st.sidebar.header('User Input Features')
        valid_values = predictor.get_valid_values()
        product = st.sidebar.selectbox('Product Purchased', valid_values['products'])
        ticket_type = st.sidebar.selectbox('Ticket Type', valid_values['ticket_types'])
        priority = st.sidebar.selectbox('Ticket Priority', valid_values['priorities'])
        status = st.sidebar.selectbox('Ticket Status', valid_values['statuses'])
        age = st.sidebar.slider('Customer Age', 18, 100, 35)

        # Main area for results
        st.header('Prediction Results')

        # Make prediction
        if st.sidebar.button('Predict'):
            try:
                prediction = predictor.predict(product, ticket_type, priority, status, age)

                # Display inputs
                st.subheader('Given Inputs')
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Product Purchased:** {prediction['details']['product']}")
                    st.markdown(f"**Ticket Type:** {prediction['details']['ticket_type']}")
                with col2:
                    st.markdown(f"**Priority:** {prediction['details']['priority']}")
                    st.markdown(f"**Status:** {prediction['details']['status']}")
                st.markdown(f"**Customer Age:** {prediction['details']['customer_age']}")

                # Display prediction and satisfaction level
                st.subheader('Prediction')
                st.markdown(f"**Predicted Satisfaction Rating:** {prediction['prediction']:.2f}")

                # Visual representation of satisfaction level
                satisfaction_level = prediction['satisfaction_level']
                st.markdown(f"**Satisfaction Level:** {satisfaction_level}")

                # Display a visual indicator for satisfaction level
                if satisfaction_level == 'Very Satisfied':
                    st.success("The customer is very satisfied!")
                elif satisfaction_level == 'Satisfied':
                    st.success("The customer is satisfied.")
                elif satisfaction_level == 'Neutral':
                    st.warning("The customer is neutral.")
                elif satisfaction_level == 'Dissatisfied':
                    st.error("The customer is dissatisfied.")
                else:
                    st.error("The customer is very dissatisfied.")

            except ValueError as e:
                st.error(str(e))
    else:
        st.info("Please upload a CSV file to proceed.")

def main():
    check_and_install_packages()

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
