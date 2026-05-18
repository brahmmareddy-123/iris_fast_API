import streamlit as st
import requests


st.title("🌸 Iris Flower Prediction")


# Inputs
sepal_length = st.slider(
    "Sepal Length",
    4.0, 8.0, 5.1
)

sepal_width = st.slider(
    "Sepal Width",
    2.0, 5.0, 3.5
)

petal_length = st.slider(
    "Petal Length",
    1.0, 7.0, 1.4
)

petal_width = st.slider(
    "Petal Width",
    0.1, 3.0, 0.2
)


if st.button("Predict"):

    url = "http://127.0.0.1:8000/predict"

    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    response = requests.post(
        url,
        json=data
    )

    result = response.json()

    st.success(
        f"Prediction: {result['predicted_species']}"
    )