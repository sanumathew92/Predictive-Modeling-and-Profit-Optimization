import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("Break-Even Commission Analysis")

# ---------------------------------------------------
# Load model
# ---------------------------------------------------

model = joblib.load(
    "models/total_profit_gb.pkl"
)

feature_names = joblib.load(
    "models/feature_names.pkl"
)

# ---------------------------------------------------
# Load data
# ---------------------------------------------------

df = pd.read_csv(
    "data/SkyCity_Preprocessed.csv"
)

# ---------------------------------------------------
# Restaurant Selector
# ---------------------------------------------------

restaurant = st.selectbox(

    "Select Restaurant",

    df["RestaurantName"].unique()

)

base_input = df[
    df["RestaurantName"] == restaurant
][feature_names].copy()

# ---------------------------------------------------
# Current Profit
# ---------------------------------------------------

current_profit = model.predict(
    base_input
)[0]

# ---------------------------------------------------
# Evaluate Different Commission Rates
# ---------------------------------------------------

commissions = np.arange(

    0.05,

    0.41,

    0.01

)

profits = []

for c in commissions:

    temp = base_input.copy()

    temp["CommissionRate"] = c

    profit = model.predict(
        temp
    )[0]

    profits.append(
        profit
    )

# ---------------------------------------------------
# Optimal Commission
# ---------------------------------------------------

profits = np.array(
    profits
)

best_index = np.argmax(
    profits
)

optimal_commission = commissions[
    best_index
]

maximum_profit = profits[
    best_index
]

# ---------------------------------------------------
# Metrics
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(

        "Current Profit",

        f"${current_profit:,.0f}"

    )

with col2:

    st.metric(

        "Optimal Commission",

        f"{optimal_commission*100:.1f}%"

    )

with col3:

    st.metric(

        "Maximum Profit",

        f"${maximum_profit:,.0f}"

    )

# ---------------------------------------------------
# Chart
# ---------------------------------------------------

st.subheader(
    "Profit vs Commission Rate"
)

fig, ax = plt.subplots(

    figsize=(8,5)

)

ax.plot(

    commissions*100,

    profits,

    linewidth=3

)

ax.axvline(

    optimal_commission*100,

    linestyle="--",

    label="Optimal"

)

ax.set_xlabel(
    "Commission Rate (%)"
)

ax.set_ylabel(
    "Profit"
)

ax.set_title(
    "Break-Even Commission Curve"
)

ax.legend()

st.pyplot(
    fig
)

# ---------------------------------------------------
# Table
# ---------------------------------------------------

results_df = pd.DataFrame({

    "Commission Rate (%)":

    commissions*100,

    "Predicted Profit":

    profits

})

st.subheader(
    "Commission Analysis Table"
)

st.dataframe(

    results_df.style.format(

        {

            "Commission Rate (%)":"{:.1f}",

            "Predicted Profit":"${:,.0f}"

        }

    )

)

# ---------------------------------------------------
# Recommendation
# ---------------------------------------------------

st.subheader(
    "Recommendation"
)

if optimal_commission < base_input["CommissionRate"].iloc[0]:

    st.success(

        "Current commission appears too high. Consider renegotiating aggregator fees."

    )

elif optimal_commission > base_input["CommissionRate"].iloc[0]:

    st.info(

        "There is room for higher commissions without reducing profitability."

    )

else:

    st.success(

        "Current commission is close to optimal."

    )