import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from scipy.optimize import minimize

# -----------------------------------------------------
# Title
# -----------------------------------------------------

st.title("Executive Recommendation Engine")

# -----------------------------------------------------
# Load Model
# -----------------------------------------------------

model = joblib.load(
    "models/total_profit_gb.pkl"
)

feature_names = joblib.load(
    "models/feature_names.pkl"
)

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------

df = pd.read_csv(
    "data/SkyCity_Preprocessed.csv"
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

# =====================================================
# Monte Carlo Simulation
# =====================================================

profits = []

for i in range(5000):

    temp = base_input.copy()

    temp["GrowthFactor"] *= np.random.uniform(
        0.98,
        1.05
    )

    temp["CommissionRate"] *= np.random.uniform(
        0.90,
        1.10
    )

    temp["DeliveryCostPerOrder"] *= np.random.uniform(
        0.80,
        1.20
    )

    temp["MonthlyOrders"] *= np.random.uniform(
        0.90,
        1.10
    )

    profit = model.predict(temp)[0]

    profits.append(
        profit
    )

profits = np.array(
    profits
)

mean_profit = profits.mean()

probability_of_loss = (

    np.mean(
        profits < 0
    )

) * 100

lower_ci = np.percentile(
    profits,
    2.5
)

upper_ci = np.percentile(
    profits,
    97.5
)

# =====================================================
# Break-Even Commission
# =====================================================

commissions = np.arange(
    0.05,
    0.41,
    0.01
)

commission_profits = []

for c in commissions:

    temp = base_input.copy()

    temp["CommissionRate"] = c

    p = model.predict(temp)[0]

    commission_profits.append(
        p
    )

best_index = np.argmax(
    commission_profits
)

optimal_commission = commissions[
    best_index
]

# =====================================================
# Channel Optimization
# =====================================================

def objective(x):

    ue = x[0]

    dd = x[1]

    sd = x[2]

    instore = max(
        0,
        1 - ue - dd - sd
    )

    temp = base_input.copy()

    temp["UE_share"] = ue

    temp["DD_share"] = dd

    temp["SD_share"] = sd

    temp["InStoreShare"] = instore

    profit = model.predict(
        temp
    )[0]

    return -profit


constraint = {

    "type":"eq",

    "fun":lambda x:

    x[0] + x[1] + x[2] - 1

}

bounds = [

    (0,1),

    (0,1),

    (0,1)

]

x0 = [

    base_input["UE_share"].iloc[0],

    base_input["DD_share"].iloc[0],

    base_input["SD_share"].iloc[0]

]

result = minimize(

    objective,

    x0,

    bounds=bounds,

    constraints=constraint

)

optimal_ue = result.x[0]

optimal_dd = result.x[1]

optimal_sd = result.x[2]

optimized_profit = -result.fun

uplift_pct = (

    optimized_profit -

    current_profit

) / current_profit * 100

# =====================================================
# KPI Dashboard
# =====================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(

        "Current Profit",

        f"${current_profit:,.0f}"

    )

with col2:

    st.metric(

        "Expected Profit",

        f"${mean_profit:,.0f}"

    )

with col3:

    st.metric(

        "Optimization Uplift",

        f"{uplift_pct:.2f}%"

    )

with col4:

    st.metric(

        "Optimal Commission",

        f"{100*optimal_commission:.1f}%"

    )

# =====================================================
# Strategic Recommendations
# =====================================================

st.header(
    "Strategic Recommendations"
)

if uplift_pct > 10:

    st.success(

        "Significant profit improvement opportunity exists through channel optimization."

    )

if optimal_sd > base_input["SD_share"].iloc[0]:

    st.success(

        "Increase investment in self-delivery operations."

    )

if optimal_commission < base_input["CommissionRate"].iloc[0]:

    st.warning(

        "Aggregator commissions should be renegotiated."

    )

if probability_of_loss < 5:

    st.success(

        "Business risk profile is low."

    )

else:

    st.warning(

        "Business exhibits moderate risk."

    )

if base_input["GrowthFactor"].iloc[0] > 1.03:

    st.success(

        "Strong demand growth opportunity exists."

    )

# =====================================================
# Optimal Channel Mix
# =====================================================

mix_df = pd.DataFrame({

    "Channel":[

        "Uber Eats",

        "DoorDash",

        "Self Delivery"

    ],

    "Optimal Share":[

        optimal_ue,

        optimal_dd,

        optimal_sd

    ]

})

st.subheader(
    "Optimal Channel Mix"
)

st.dataframe(

    mix_df.style.format(

        {

            "Optimal Share":"{:.2%}"

        }

    )

)

# =====================================================
# Risk Assessment
# =====================================================

st.subheader(
    "Risk Assessment"
)

st.write(

f"""

Expected Monthly Profit:

${mean_profit:,.0f}

95% Confidence Interval:

${lower_ci:,.0f} to ${upper_ci:,.0f}

Probability of Loss:

{probability_of_loss:.2f}%

"""

)

# =====================================================
# Profit Distribution
# =====================================================

st.subheader(
    "Monte Carlo Profit Distribution"
)

fig, ax = plt.subplots(

    figsize=(8,5)

)

ax.hist(

    profits,

    bins=40

)

ax.set_xlabel(
    "Profit"
)

ax.set_ylabel(
    "Frequency"
)

ax.set_title(
    "Profit Distribution"
)

st.pyplot(
    fig
)