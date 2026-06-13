import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from utilities.preprocessing import create_features

st.title("Profit Prediction")

# ---------------------------------------------------
# Base Directory
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------
# Load model
# ---------------------------------------------------

model = joblib.load(
    BASE_DIR / "Models" / "total_profit_gb.pkl"
)

# ---------------------------------------------------
# Load average restaurant values
# ---------------------------------------------------

input_df = pd.read_csv(
    BASE_DIR / "data" / "mean_input.csv"
)

# ---------------------------------------------------
# User Inputs
# ---------------------------------------------------

monthly_orders = st.slider(
    "Monthly Orders",
    1000,
    20000,
    int(input_df["MonthlyOrders"][0])
)

aov = st.slider(
    "Average Order Value",
    20.0,
    60.0,
    float(input_df["AOV"][0])
)

growth_factor = st.slider(
    "Growth Factor",
    0.95,
    1.10,
    float(input_df["GrowthFactor"][0])
)

commission_rate = st.slider(
    "Commission Rate",
    0.05,
    0.40,
    float(input_df["CommissionRate"][0])
)

delivery_cost = st.slider(
    "Delivery Cost Per Order",
    0.5,
    10.0,
    float(input_df["DeliveryCostPerOrder"][0])
)

delivery_radius = st.slider(
    "Delivery Radius (KM)",
    3,
    20,
    int(input_df["DeliveryRadiusKM"][0])
)

UE_share = st.slider(
    "Uber Eats Share",
    0.0,
    1.0,
    float(input_df["UE_share"][0])
)

DD_share = st.slider(
    "DoorDash Share",
    0.0,
    1.0,
    float(input_df["DD_share"][0])
)

SD_share = st.slider(
    "Self Delivery Share",
    0.0,
    1.0,
    float(input_df["SD_share"][0])
)

# ---------------------------------------------------
# Update Inputs
# ---------------------------------------------------

input_df["MonthlyOrders"] = monthly_orders
input_df["AOV"] = aov
input_df["GrowthFactor"] = growth_factor
input_df["CommissionRate"] = commission_rate
input_df["DeliveryCostPerOrder"] = delivery_cost
input_df["DeliveryRadiusKM"] = delivery_radius
input_df["UE_share"] = UE_share
input_df["DD_share"] = DD_share
input_df["SD_share"] = SD_share

# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------

input_df = create_features(input_df)

# ---------------------------------------------------
# Load feature names
# ---------------------------------------------------

feature_names = joblib.load(
    BASE_DIR / "Models" / "feature_names.pkl"
)

# Select only model features
input_df = input_df[feature_names]

# ---------------------------------------------------
# Predict Profit
# ---------------------------------------------------

if st.button("Predict Profit"):

    prediction = model.predict(input_df)

    st.success(
        f"Predicted Monthly Profit = ${prediction[0]:,.2f}"
    )
