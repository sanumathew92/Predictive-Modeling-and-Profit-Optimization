import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Title
# -----------------------------------------------------

st.title("Monte Carlo Simulation")

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
# Number of simulations
# -----------------------------------------------------

n_simulations = st.slider(

    "Number of Simulations",

    1000,

    20000,

    10000,

    step=1000

)

# -----------------------------------------------------
# Monte Carlo Simulation
# -----------------------------------------------------

profits = []

for i in range(n_simulations):

    temp = base_input.copy()

    # Demand uncertainty
    temp["GrowthFactor"] *= np.random.uniform(
        0.98,
        1.05
    )

    # Commission uncertainty
    temp["CommissionRate"] *= np.random.uniform(
        0.90,
        1.10
    )

    # Delivery cost uncertainty
    temp["DeliveryCostPerOrder"] *= np.random.uniform(
        0.80,
        1.20
    )

    # Monthly orders uncertainty
    temp["MonthlyOrders"] *= np.random.uniform(
        0.90,
        1.10
    )

    profit = model.predict(
        temp
    )[0]

    profits.append(
        profit
    )

profits = np.array(
    profits
)

# -----------------------------------------------------
# Metrics
# -----------------------------------------------------

mean_profit = profits.mean()

std_profit = profits.std()

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

# -----------------------------------------------------
# KPIs
# -----------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(

        "Expected Profit",

        f"${mean_profit:,.0f}"

    )

with col2:

    st.metric(

        "Std Deviation",

        f"${std_profit:,.0f}"

    )

with col3:

    st.metric(

        "Probability of Loss",

        f"{probability_of_loss:.2f}%"

    )

with col4:

    st.metric(

        "95% Confidence Interval",

        f"${lower_ci:,.0f} - ${upper_ci:,.0f}"

    )

# -----------------------------------------------------
# Histogram
# -----------------------------------------------------

st.subheader(
    "Profit Distribution"
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
    "Monte Carlo Profit Distribution"
)

st.pyplot(
    fig
)

# -----------------------------------------------------
# Risk Assessment
# -----------------------------------------------------

st.subheader(
    "Risk Assessment"
)

if probability_of_loss < 5:

    st.success(

        "Low Risk Business"

    )

elif probability_of_loss < 15:

    st.warning(

        "Moderate Risk Business"

    )

else:

    st.error(

        "High Risk Business"

    )

# -----------------------------------------------------
# Summary
# -----------------------------------------------------

st.subheader(
    "Simulation Summary"
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