import pandas as pd
import numpy as np
import random
from faker import Faker
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(0)

# Create a fake data generator
fake = Faker()

# Generate disaster data with separate latitude and longitude columns
disaster_data = []
for i in range(100):
    disaster_id = i + 1
    disaster_type = random.choice(['earthquake', 'flood', 'hurricane'])
    latlng = fake.latlng()  # Get the tuple
    latitude = latlng[0]  # Extract latitude
    longitude = latlng[1]  # Extract longitude
    disaster_severity = random.randint(1, 10)
    casualties = np.random.randint(0, 1000)  # Fake casualty data
    area_impacted_sqft = np.random.randint(10000, 200000)
    
    # Calculate the budget allocation based on severity, casualties, and area_impacted_sqft
    budget_increase = 123 * (disaster_severity + casualties + area_impacted_sqft)
    budget_allocation = max(budget_increase, 0)  # Ensure budget_allocation is non-negative
    
    disaster_data.append([disaster_id, disaster_type, latitude, longitude, disaster_severity, budget_allocation, casualties, area_impacted_sqft])

# Create a DataFrame from the generated data
disaster_df = pd.DataFrame(disaster_data, columns=['disaster_id', 'disaster_type', 'latitude', 'longitude', 'disaster_severity', 'budget_allocation', 'casualties', 'area_impacted_sqft'])

# Convert 'latitude' and 'longitude' columns to numeric (float) data types
disaster_df['latitude'] = pd.to_numeric(disaster_df['latitude'], errors='coerce')
disaster_df['longitude'] = pd.to_numeric(disaster_df['longitude'], errors='coerce')

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Apply label encoding to 'disaster_type'
disaster_df['disaster_type_encoded'] = label_encoder.fit_transform(disaster_df['disaster_type'])

# Define features (X) and target variable (y)
X = disaster_df[['disaster_type_encoded', 'casualties', 'area_impacted_sqft', 'disaster_severity']]
y = disaster_df['budget_allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model and train it on the training data
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Define a function to predict budget allocation
def predict_budget(disaster_type_encoded, casualties, area_impacted_sqft, disaster_severity):
    sample_data_point = [[disaster_type_encoded, casualties, area_impacted_sqft, disaster_severity]]
    prediction = linear_model.predict(sample_data_point)
    return prediction[0]

# Streamlit app
def main():
    st.title("Disaster Relief Fund Allocation")
    st.write("""

Developed by: Alpha Diallo

Summary:

The Disaster Relief Fund Allocation project is a critical initiative aimed at efficiently distributing financial resources to communities affected by natural disasters. It involves a comprehensive approach to determine the allocation of funds based on various factors, such as the scale of the disaster. This project ensures that the available financial aid is distributed equitably and effectively to provide timely support to affected individuals and regions.

Description:

Natural disasters, such as earthquakes, hurricanes, and floods, can have devastating effects on communities, resulting in loss of lives and extensive damage to infrastructure. To address these challenges, the Disaster Relief Fund Allocation project has been developed to streamline the process of allocating financial resources for disaster relief efforts.

Key Components:

Disaster Assessment: The project begins with a thorough assessment of the disaster's impact, including the number of lives lost, the extent of the affected area, and the severity of the disaster. This assessment is crucial for understanding the financial budget allocation.


Timely Allocation: The project focuses on expeditious allocation of funds to provide immediate relief to those in need.


Key Benefits:

 - Efficient Resource Distribution

 - Equitable Support

 - Timely Assistance

 - Data-Driven Decision-Making
""")
             
    # Input fields
    disaster_type_encoded = st.selectbox("Select Disaster Type", ["earthquake", "flood", "hurricane"])
    casualties = st.slider("Casualties", min_value=0, max_value=1000, value=500)
    area_impacted_sqft = st.slider("Area Impacted (sqft)", min_value=10000, max_value=200000, value=100000)
    disaster_severity = st.slider("Disaster Severity", min_value=1, max_value=10, value=5)

    # Convert the selected disaster type to its encoded value
    disaster_type_encoded = {"earthquake": 0, "flood": 1, "hurricane": 2}[disaster_type_encoded]

    if st.button("Predict Budget Allocation"):
        budget_allocation = predict_budget(disaster_type_encoded, casualties, area_impacted_sqft, disaster_severity)
        st.success(f"Predicted Budget Allocation: ${budget_allocation:.2f}")

if __name__ == "__main__":
    main()
