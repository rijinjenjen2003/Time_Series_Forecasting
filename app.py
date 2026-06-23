import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.title("📈 Sales Forecast App")

# Load Data
df = pd.read_csv("train.csv")

df["date"] = pd.to_datetime(df["date"])

daily_sales = (
    df.groupby("date")["sales"]
    .sum()
    .reset_index()
)

daily_sales.columns = ["ds", "y"]

# Train Model
model = Prophet()
model.fit(daily_sales)

# Slider
days = st.slider(
    "Forecast Days",
    7,
    90,
    30
)

# Future Dates
future = model.make_future_dataframe(
    periods=days
)

# Prediction
forecast = model.predict(future)

# Small Graph
st.subheader("📈 Forecast Graph")

fig, ax = plt.subplots(figsize=(6,3))

ax.plot(
    forecast["ds"].tail(days),
    forecast["yhat"].tail(days)
)

ax.set_title("Future Sales")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")

st.pyplot(fig)

# Table
st.subheader("Forecast Results")

st.dataframe(
    forecast[
        ["ds", "yhat"]
    ].tail(days)
)
st.pyplot(fig)
