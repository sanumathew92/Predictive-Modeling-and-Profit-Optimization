import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------
# Title
# -----------------------------------------------------

st.title("What-If Simulator")

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
# Load Dataset
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

# -----------------------------------------------------
# User Inputs
# -----------------------------------------------------

growth_factor = st.slider(
    "Growth Factor",
    0.90,
    1.10,
    float(base_input["GrowthFactor"].iloc[0])
)

monthly_orders = st.slider(
    "Monthly Orders",
    100,
    5000,
    int(base_input["MonthlyOrders"].iloc[0])
)

aov = st.slider(
    "Average Order Value",
    10.0,
    100.0,
    float(base_input["AOV"].iloc[0])
)

commission_rate = st.slider(
    "Commission Rate",
    0.05,
    0.40,
    float(base_input["CommissionRate"].iloc[0])
)

delivery_cost = st.slider(
    "Delivery Cost Per Order",
    1.0,
    10.0,
    float(base_input["DeliveryCostPerOrder"].iloc[0])
)

ue_share = st.slider(
    "Uber Eats Share",
    0.0,
    1.0,
    float(base_input["UE_share"].iloc[0])
)

dd_share = st.slider(
    "DoorDash Share",
    0.0,
    1.0,
    float(base_input["DD_share"].iloc[0])
)

sd_share = st.slider(
    "Self Delivery Share",
    0.0,
    1.0,
    float(base_input["SD_share"].iloc[0])
)

# -----------------------------------------------------
# Scenario
# -----------------------------------------------------

scenario = base_input.copy()

scenario["GrowthFactor"] = growth_factor
scenario["MonthlyOrders"] = monthly_orders
scenario["AOV"] = aov
scenario["CommissionRate"] = commission_rate
scenario["DeliveryCostPerOrder"] = delivery_cost

scenario["UE_share"] = ue_share
scenario["DD_share"] = dd_share
scenario["SD_share"] = sd_share

scenario["InStoreShare"] = round(
    max(
        0,
        1 - ue_share - dd_share - sd_share
    ),
    4
)

scenario = scenario[feature_names]

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

scenario_profit = model.predict(
    scenario
)[0]

# -----------------------------------------------------
# Profit Change
# -----------------------------------------------------

profit_change = (
    scenario_profit -
    current_profit
)

profit_change_pct = (
    profit_change /
    current_profit
) * 100

# -----------------------------------------------------
# Metrics
# -----------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Current Profit",
        f"${current_profit:,.0f}"
    )

with col2:

    st.metric(
        "Scenario Profit",
        f"${scenario_profit:,.0f}"
    )

with col3:

    st.metric(
        "Profit Change",
        f"${profit_change:,.0f}"
    )

with col4:

    st.metric(
        "Profit Change %",
        f"{profit_change_pct:.2f}%"
    )

# -----------------------------------------------------
# Interpretation
# -----------------------------------------------------

st.subheader(
    "Business Interpretation"
)

if profit_change_pct > 10:

    st.success(
        "High upside opportunity detected."
    )

elif profit_change_pct > 0:

    st.info(
        "Moderate improvement scenario."
    )

else:

    st.warning(
        "Scenario reduces profitability."
    )

# -----------------------------------------------------
# Scenario Table
# -----------------------------------------------------

st.subheader(
    "Scenario Inputs"
)

st.dataframe(
    scenario.T.round(4)
)