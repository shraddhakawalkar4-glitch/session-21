from sklearn.preprocessing import StandardScaler
#Question 01
# Streamlit is used to build interactive web apps directly from Python scripts
import streamlit as st       

# Pandas is used for data manipulation and analysis, especially with tabular datasets
import pandas as pd

# Joblib is used to save and load machine learning models efficiently
import joblib


#Q2. (Loading Model and Preprocessing Objects)
model = joblib.load('./LR_model.pkl')
scaler = joblib.load('./scaler.pkl')
encoded_columns = joblib.load('./columns.pkl')

#Q3. (Page Configuration)
st.set_page_config(
    page_title = "Ford Car Price Predictor",
    layout = "centered", 
)
# Configure Streamlit page settings
# Title shown in browser tab
# Keeps content neatly centered on the page

#Q4. (Title and Description)
st.title("Ford Car Price")
st.write("Enter the car details below to predict its selling price.")

#Q5. (Numerical Input Fields)
year = st.number_input(
    "Manufacturing Year",
    min_value = 1995,
    max_value = 2025,
    value = 2018
)

mileage = st.number_input(
    "Mileage",
    min_value = 1,
    max_value = 180000,
    value = 1000
)

tax = st.number_input(
    "Road Tax",
    min_value = 0,
    max_value = 250,
    value = 50
)

mpg = st.number_input(
    "MPG",
    min_value = 1,
    max_value = 100,
    value = 20
)

engine = st.number_input(
    "Engine Size",
    min_value = 1,
    max_value = 5,
    value = 1
)

# Q6. (Categorical Input using Dropdowns)

# Advantage of selectbox:
# User-friendly dropdown
# Prevents wrong/invalid input
# Keeps data clean for prediction

transmission = st.selectbox(
    "Transmission",
    [
       "Automatic",
       "Manual",
       "Semi-Auto" 
    ]
)

fuelType = st.selectbox(
    "Fuel Type",
    [
       "Petrol",
       "Diesel",
       "Hybrid",
       "Electric",
        "Other"
    ]
)

#Q7. (Text Input and Predict Button)
model_name = st.text_input(
    "Model","Focus"
)

#Q8. (Creating Input DataFrame & Encoding)
if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "model": [model_name],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuelType],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine]
    })
    print(input_df)

    encoded_input_df = pd.get_dummies(input_df).astype(int)
    encoded_input_df = encoded_input_df.reindex(columns=encoded_columns, fill_value=0)
    print(encoded_columns)
    

 #Q9. (Feature Scaling and Prediction)
    numeric_col = ["year",  "mileage",  "tax",  "mpg",  "engineSize"]
    scaler = StandardScaler()
    encoded_input_df[numeric_col] = scaler.fit_transform(encoded_input_df[numeric_col])
    print(encoded_input_df)

    predicted = model.predict(encoded_input_df)
    st.success(f"Predicted Price: £{round(predicted[0], 2)}")