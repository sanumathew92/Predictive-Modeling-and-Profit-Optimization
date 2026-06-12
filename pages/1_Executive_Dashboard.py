import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Executive Dashboard",
    layout="wide"
)

st.title("📊 Executive Dashboard")

# =====================================================
# Load Data
# =====================================================

df = pd.read_csv(
    "data\SkyCity_Feature_Engineered.csv"
)


# =====================================================
# KPI Metrics
# =====================================================

total_revenue = df["TotalRevenue"].sum()

total_profit = df["TotalNetProfit"].sum()

total_orders = df["MonthlyOrders"].sum()

profit_margin = (
    total_profit /
    total_revenue
) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Revenue",
        f"${total_revenue:,.0f}"
    )

with col2:
    st.metric(
        "Profit",
        f"${total_profit:,.0f}"
    )

with col3:
    st.metric(
        "Orders",
        f"{total_orders:,.0f}"
    )

with col4:
    st.metric(
        "Profit Margin",
        f"{profit_margin:.2f}%"
    )

st.divider()

# =====================================================
# Revenue by Channel
# =====================================================

col1, col2 = st.columns(2)

with col1:

    revenue_channel = {

        "InStore":
        df["InStoreRevenue"].sum(),

        "Uber Eats":
        df["UberEatsRevenue"].sum(),

        "DoorDash":
        df["DoorDashRevenue"].sum(),

        "Self Delivery":
        df["SelfDeliveryRevenue"].sum()

    }

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(
        revenue_channel.keys(),
        revenue_channel.values()
    )

    ax.set_title(
        "Revenue by Channel"
    )

    ax.set_ylabel(
        "Revenue"
    )

    st.pyplot(fig)

# =====================================================
# Profit by Channel
# =====================================================

with col2:

    profit_channel = {

        "InStore":
        df["InStoreNetProfit"].sum(),

        "Uber Eats":
        df["UberEatsNetProfit"].sum(),

        "DoorDash":
        df["DoorDashNetProfit"].sum(),

        "Self Delivery":
        df["SelfDeliveryNetProfit"].sum()

    }

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(
        profit_channel.keys(),
        profit_channel.values()
    )

    ax.set_title(
        "Profit by Channel"
    )

    ax.set_ylabel(
        "Profit"
    )

    st.pyplot(fig)

st.divider()

# =====================================================
# Channel Share Pie Chart
# =====================================================

col1, col2 = st.columns(2)

with col1:

    channel_share = [

        df["InStoreShare"].mean(),

        df["UE_share"].mean(),

        df["DD_share"].mean(),

        df["SD_share"].mean()

    ]

    labels = [

        "InStore",

        "Uber Eats",

        "DoorDash",

        "Self Delivery"

    ]

    fig, ax = plt.subplots()

    ax.pie(

        channel_share,

        labels=labels,

        autopct="%1.1f%%"

    )

    ax.set_title(
        "Average Channel Mix"
    )

    st.pyplot(fig)

# =====================================================
# Segment Analysis
# =====================================================

with col2:

    segment_counts = (
        df["Segment"]
        .value_counts()
    )

    fig, ax = plt.subplots()

    ax.bar(

        segment_counts.index,

        segment_counts.values

    )

    ax.set_title(
        "Segment Distribution"
    )

    plt.xticks(rotation=20)

    st.pyplot(fig)

st.divider()

# =====================================================
# Subregion Analysis
# =====================================================

subregion_counts = (

    df["Subregion"]

    .value_counts()

)

fig, ax = plt.subplots(figsize=(7,4))

ax.bar(

    subregion_counts.index,

    subregion_counts.values

)

ax.set_title(
    "Restaurant Distribution by Subregion"
)

plt.xticks(rotation=20)

st.pyplot(fig)

st.divider()

# =====================================================
# Cuisine Distribution
# =====================================================

st.subheader(
    "Cuisine Distribution"
)

cuisine_counts = (

    df["CuisineType"]

    .value_counts()

)

fig, ax = plt.subplots(

    figsize=(8,4)

)

ax.bar(

    cuisine_counts.index,

    cuisine_counts.values

)

ax.set_title(

    "Cuisine Distribution"

)

plt.xticks(rotation=45)

st.pyplot(fig)

# =====================================================
# Top 10 Restaurants
# =====================================================

st.subheader(
    "Top 10 Restaurants by Profit"
)

top10 = (

    df

    .groupby("RestaurantName")["TotalNetProfit"]

    .sum()

    .sort_values(ascending=False)

    .head(10)

)

st.dataframe(top10)

# =====================================================
# Executive Insights
# =====================================================

st.subheader(
    "Executive Insights"
)

best_channel = max(

    profit_channel,

    key=profit_channel.get

)

st.success(

    f"Highest overall profit contribution comes from {best_channel}."

)

st.success(

    f"Average Profit Margin = {profit_margin:.2f}%."

)

st.success(

    f"Monthly order volume across all restaurants = {total_orders:,.0f}."

)

st.success(

    "Self-delivery and aggregator mix should be continuously optimized using the scenario engine."

)