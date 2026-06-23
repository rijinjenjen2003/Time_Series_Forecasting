import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.title("📈 Sales Forecast App")

# Load Saved Model
with open("prophet_model.pkl", "rb") as f:
    model = pickle.load(f)

st.success("Model Loaded Successfully")

# Forecast Days
days = st.slider(
    "Forecast Days",
    7,
    90,
    30
)

# Create Future Dates
future = model.make_future_dataframe(
    periods=days
)

# Predict
forecast = model.predict(future)

# Graph
st.subheader("📈 Forecast Graph")

fig, ax = plt.subplots(figsize=(6, 3))

ax.plot(
    forecast["ds"].tail(days),
    forecast["yhat"].tail(days)
)

ax.set_title("Future Sales Forecast")

st.pyplot(fig)

# Table
st.subheader("Forecast Results")

st.dataframe(
    forecast[
        ["ds", "yhat"]
    ].tail(days)
)
st.pyplot(fig)
