import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# Title
st.title("📈 Sales Forecast App")

# Load Model
with open("model/prophet_model.pkl", "rb") as f:
    model = pickle.load(f)

st.success("Model Loaded Successfully")

# Forecast Days
days = st.slider(
    "Select Forecast Days",
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

# Show Results
st.subheader("Forecast Results")

st.dataframe(
    forecast[
        ["ds", "yhat"]
    ].tail(days)
)

st.subheader("📈 Actual vs Forecast")

fig, ax = plt.subplots(figsize=(12, 5))

# Forecast
ax.plot(
    forecast["ds"],
    forecast["yhat"],
    label="Forecast"
)

ax.set_title("Sales Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)