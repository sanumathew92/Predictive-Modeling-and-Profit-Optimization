import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Title
# -----------------------------------------------------

st.title("Sensitivity Analysis")

# -----------------------------------------------------
# Load model
# -----------------------------------------------------

model = joblib.load(
    "models/total_profit_gb.pkl"
)

feature_names = joblib.load(
    "models/feature_names.pkl"
)

# -----------------------------------------------------
# Load data
# -----------------------------------------------------

df = pd.read_csv(
    "data/SkyCity_Preprocessed.csv"
)

# -----------------------------------------------------
# Restaurant selector
# -----------------------------------------------------

restaurant = st.selectbox(

    "Select Restaurant",

    df["RestaurantName"].unique()

)

base_input = df[
    df["RestaurantName"] == restaurant
][feature_names].copy()

# -----------------------------------------------------
# Current profit
# -----------------------------------------------------

current_profit = model.predict(
    base_input
)[0]

st.metric(

    "Current Profit",

    f"${current_profit:,.0f}"

)

# -----------------------------------------------------
# Variables to test
# -----------------------------------------------------

variables = [

    "MonthlyOrders",

    "AOV",

    "GrowthFactor",

    "CommissionRate",

    "DeliveryCostPerOrder",

    "DeliveryRadiusKM",

    "UE_share",

    "DD_share",

    "SD_share"

]

# -----------------------------------------------------
# Sensitivity
# -----------------------------------------------------

results = []

for var in variables:

    high = base_input.copy()

    low = base_input.copy()

    high[var] *= 1.10

    low[var] *= 0.90

    profit_high = model.predict(
        high
    )[0]

    profit_low = model.predict(
        low
    )[0]

    impact = profit_high - profit_low

    results.append(

        [

            var,

            impact

        ]

    )

# -----------------------------------------------------
# Sensitivity dataframe
# -----------------------------------------------------

sensitivity_df = pd.DataFrame(

    results,

    columns=[

        "Variable",

        "Impact"

    ]

)

sensitivity_df = sensitivity_df.sort_values(

    "Impact",

    ascending=True

)

# -----------------------------------------------------
# Table
# -----------------------------------------------------

st.subheader(
    "Sensitivity Table"
)

st.dataframe(

    sensitivity_df.style.format(

        {

            "Impact":"{:,.0f}"

        }

    )

)

# -----------------------------------------------------
# Tornado Chart
# -----------------------------------------------------

st.subheader(
    "Tornado Chart"
)

fig, ax = plt.subplots(

    figsize=(8,6)

)

ax.barh(

    sensitivity_df["Variable"],

    sensitivity_df["Impact"]

)

ax.set_xlabel(

    "Profit Impact"

)

ax.set_title(

    "Profit Sensitivity"

)

st.pyplot(
    fig
)

# -----------------------------------------------------
# Profit Sensitivity Index
# -----------------------------------------------------

psi_results = []

for var in variables:

    temp = base_input.copy()

    temp[var] *= 1.10

    new_profit = model.predict(
        temp
    )[0]

    psi = (

        abs(

            new_profit -

            current_profit

        )

        /

        current_profit

    ) * 100

    psi_results.append(

        [

            var,

            psi

        ]

    )

psi_df = pd.DataFrame(

    psi_results,

    columns=[

        "Variable",

        "PSI"

    ]

)

psi_df = psi_df.sort_values(

    "PSI",

    ascending=False

)

# -----------------------------------------------------
# PSI Table
# -----------------------------------------------------

st.subheader(
    "Profit Sensitivity Index"
)

st.dataframe(

    psi_df.style.format(

        {

            "PSI":"{:.2f}%"

        }

    )

)

# -----------------------------------------------------
# Insight
# -----------------------------------------------------

most_sensitive = psi_df.iloc[0]["Variable"]

st.subheader(
    "Business Insight"
)

st.success(

    f"Profit is most sensitive to {most_sensitive}."

)