import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Page Configuration

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)


st.title("🚗 Car Price Prediction using Machine Learning")

st.write(
    "Upload your car dataset CSV and predict car prices."
)


# Upload CSV

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)



if uploaded_file:

    df = pd.read_csv(uploaded_file)


else:

    st.warning(
        "No CSV uploaded. Using sample car dataset."
    )


    df = pd.DataFrame({

        "brand":
        [
            "Toyota",
            "BMW",
            "Honda",
            "Audi",
            "Hyundai",
            "Ford",
            "Tesla",
            "Kia",
            "Mercedes",
            "Tata"
        ],

        "year":
        [
            2020,2019,2021,2020,2022,
            2021,2023,2022,2020,2021
        ],

        "horsepower":
        [
            120,200,100,180,110,
            130,250,140,220,115
        ],

        "mileage":
        [
            18,12,20,14,19,
            17,15,18,13,21
        ],

        "engine_size":
        [
            1500,2000,1200,1800,1400,
            1600,2500,1700,2200,1300
        ],

        "price":
        [
            850000,
            2500000,
            700000,
            2200000,
            800000,
            950000,
            4500000,
            1100000,
            3500000,
            750000
        ]

    })



st.subheader("Dataset Preview")

st.dataframe(df)



# Missing values

df = df.dropna()



# Encoding

encoder = LabelEncoder()


for col in df.select_dtypes(include=["object","string"]).columns:

    df[col] = encoder.fit_transform(df[col])



# Find price column

price_col = None


for col in df.columns:

    if "price" in col.lower():

        price_col = col
        break



if price_col is None:

    st.error(
        "Price column not found. Rename price column as price"
    )

    st.stop()



X = df.drop(price_col,axis=1)

y = df[price_col]



# Train Test Split


X_train,X_test,y_train,y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)



# Model


model = LinearRegression()


model.fit(
    X_train,
    y_train
)



prediction = model.predict(X_test)



# Evaluation


mae = mean_absolute_error(
    y_test,
    prediction
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction
    )
)


r2 = r2_score(
    y_test,
    prediction
)



st.subheader("📊 Model Performance")


col1,col2,col3 = st.columns(3)


col1.metric(
    "MAE",
    round(mae,2)
)


col2.metric(
    "RMSE",
    round(rmse,2)
)


col3.metric(
    "R2 Score",
    round(r2,3)
)



# Graph


st.subheader(
    "Actual vs Predicted Price"
)


fig,ax = plt.subplots()


ax.scatter(
    y_test,
    prediction
)


ax.set_xlabel(
    "Actual Price"
)


ax.set_ylabel(
    "Predicted Price"
)


ax.set_title(
    "Car Price Prediction"
)


st.pyplot(fig)



# Prediction section


st.subheader(
    "🚘 Predict New Car Price"
)



user_input={}



for col in X.columns:

    user_input[col]=st.number_input(

        f"Enter {col}",

        value=float(X[col].mean())

    )



input_df=pd.DataFrame(
    [user_input]
)



if st.button("Predict Price"):


    result=model.predict(
        input_df
    )


    st.success(

        f"Estimated Car Price : ₹ {round(result[0])}"

    )