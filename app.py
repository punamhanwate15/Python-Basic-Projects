#question-1

# Import the required libraries
# Import Streamlit to build the web application interface
import streamlit as st

# Import Pandas for creating and handling DataFrames
import pandas as pd

# Import Joblib to load the trained model and saved files
import joblib
from sklearn.preprocessing import StandardScaler


#question-2

# Load the trained machine learning model and preprocessing files
model = joblib.load("./LR_ford_car.pkl")
scaler = joblib.load("./scaler.pkl")
encoded_columns = joblib.load("./columns.pkl")


#question-3

# Configure the Streamlit application
# Set the page title and keep the layout centered
st.set_page_config(
    page_title="ford car price predictor",
    layout="centered"
)


#question-4

# Display application title and instructions
st.title("Ford Car Price Prediction")
st.write("Enter the car details below to predict its selling price.")


#question-5

# Take numerical inputs from the user
year = st.number_input(
    "Manufacturing year",
    min_value=1995,
    max_value=2024,
    value=2020
)

mileage = st.number_input(
    "Mileage",
    min_value=5000,
    max_value=150000,
    value=25000
)

tax = st.number_input(
    "Road tax",
    min_value=0,
    max_value=300,
    value=145
)

mpg = st.number_input(
    "MPG",
    min_value=20,
    max_value=80,
    value=50
)

engineSize = st.number_input(
    "Engine Size",
    min_value=1,
    max_value=5,
    value=2
)


#question-6

# Select categorical values using dropdown menus
transmission = st.selectbox(
    "Transmission",
    [
        "Automatic",
        "Manual",
        "Semi-Auto"
    ]
)

fuelType = st.selectbox(
    "FuelType",
    [
        "Petrol",
        "Diesel",
        "Hybrid"
    ]
)


#question-7

# Get the model name from the user
model_name = st.text_input("Model", "Focus")

# Create a button to predict the car price
if st.button("Predict Price"):

    # Create a DataFrame using user inputs

    #question-8

    input_df = pd.DataFrame({
        "year": [year],
        "model": [model_name],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuelType],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engineSize]
    })

    print(input_df)

    # Convert categorical values into encoded format
    encoded_input_df = pd.get_dummies(input_df).astype(int)
    encoded_input_df = encoded_input_df.reindex(columns=encoded_columns, fill_value=0)

    print(encoded_columns)


    #question-9

    # Apply feature scaling to numerical columns
    numeric_col = ["year", "mileage", "tax", "mpg", "engineSize"]
    scaler = StandardScaler()
    encoded_input_df[numeric_col] = scaler.fit_transform(encoded_input_df[numeric_col])

    print(encoded_input_df)

    # Predict the car price using the trained model
    predicted = model.predict(encoded_input_df)

    # Display the predicted price
    st.success(f"Predicted Price: £{round(predicted[0], 2)}")

