import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


from scipy.optimize import minimize

# -----------------------------------------------------
# Title
# -----------------------------------------------------

st.title("Channel Mix Optimization")

# ---------------------------------------------------
# Base Directory
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# -----------------------------------------------------
# Load Model
# -----------------------------------------------------

model = joblib.load(
    BASE_DIR / "Models" / "total_profit_gb.pkl"
)

feature_names = joblib.load(
    BASE_DIR / "Models" / "feature_names.pkl"
)

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------

df = pd.read_csv(
    BASE_DIR / "data" / "SkyCity_Preprocessed.csv"
)
# -----------------------------------------------------
# Restaurant Selector
# -----------------------------------------------------

restaurant = st.selectbox(

    "Select Restaurant",

    df["RestaurantName"].unique()

)

base_input = df[
    df["RestaurantName"] == restaurant
][feature_names].copy()

# -----------------------------------------------------
# Current Profit
# -----------------------------------------------------

current_profit = model.predict(
    base_input
)[0]

# -----------------------------------------------------
# Objective Function
# -----------------------------------------------------

def objective(x):

    ue_share = x[0]

    dd_share = x[1]

    sd_share = x[2]

    instore_share = max(
        0,
        1 - ue_share - dd_share - sd_share
    )

    temp = base_input.copy()

    temp["UE_share"] = ue_share

    temp["DD_share"] = dd_share

    temp["SD_share"] = sd_share

    temp["InStoreShare"] = instore_share

    temp = temp[feature_names]

    profit = model.predict(
        temp
    )[0]

    return -profit


# -----------------------------------------------------
# Constraint
# -----------------------------------------------------

constraint = {

    "type":"eq",

    "fun":lambda x:

    x[0] + x[1] + x[2] - 1

}

# -----------------------------------------------------
# Bounds
# -----------------------------------------------------

bounds = [

    (0,1),

    (0,1),

    (0,1)

]

# -----------------------------------------------------
# Initial Values
# -----------------------------------------------------

x0 = [

    base_input["UE_share"].iloc[0],

    base_input["DD_share"].iloc[0],

    base_input["SD_share"].iloc[0]

]

# -----------------------------------------------------
# Optimization
# -----------------------------------------------------

result = minimize(

    objective,

    x0,

    bounds=bounds,

    constraints=constraint

)

# -----------------------------------------------------
# Optimal Shares
# -----------------------------------------------------

optimal_ue = result.x[0]

optimal_dd = result.x[1]

optimal_sd = result.x[2]

optimized_profit = -result.fun

uplift_pct = (

    optimized_profit -

    current_profit

) / current_profit * 100

# -----------------------------------------------------
# KPIs
# -----------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(

        "Current Profit",

        f"${current_profit:,.0f}"

    )

with col2:

    st.metric(

        "Optimized Profit",

        f"${optimized_profit:,.0f}"

    )

with col3:

    st.metric(

        "Profit Uplift",

        f"{uplift_pct:.2f}%"

    )

# -----------------------------------------------------
# Mix Comparison Table
# -----------------------------------------------------

current_mix = pd.DataFrame({

    "Channel":[

        "Uber Eats",

        "DoorDash",

        "Self Delivery"

    ],

    "Current Share":[

        base_input["UE_share"].iloc[0],

        base_input["DD_share"].iloc[0],

        base_input["SD_share"].iloc[0]

    ],

    "Optimal Share":[

        optimal_ue,

        optimal_dd,

        optimal_sd

    ]

})

st.subheader(
    "Current vs Optimal Mix"
)

st.dataframe(

    current_mix.style.format(

        {

            "Current Share":"{:.2%}",

            "Optimal Share":"{:.2%}"

        }

    )

)

# -----------------------------------------------------
# Bar Chart
# -----------------------------------------------------

fig, ax = plt.subplots(

    figsize=(8,5)

)

x = np.arange(3)

width = 0.3

ax.bar(

    x - width/2,

    current_mix["Current Share"],

    width,

    label="Current"

)

ax.bar(

    x + width/2,

    current_mix["Optimal Share"],

    width,

    label="Optimal"

)

ax.set_xticks(x)

ax.set_xticklabels(

    current_mix["Channel"]

)

ax.set_ylabel(

    "Share"

)

ax.set_title(

    "Channel Mix Optimization"

)

ax.legend()

st.pyplot(
    fig
)

# -----------------------------------------------------
# Recommendations
# -----------------------------------------------------

st.subheader(
    "Recommendations"
)

if optimal_sd > base_input["SD_share"].iloc[0]:

    st.success(

        "Increase investment in self-delivery."

    )

if optimal_ue < base_input["UE_share"].iloc[0]:

    st.warning(

        "Reduce dependency on Uber Eats."

    )

if optimal_dd < base_input["DD_share"].iloc[0]:

    st.warning(

        "Reduce DoorDash exposure."

    )

if uplift_pct > 10:

    st.success(

        "Significant optimization opportunity exists."

    )