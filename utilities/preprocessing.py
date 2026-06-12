import pandas as pd

def create_features(df):

    df["Commission_UE"] = (
        df["CommissionRate"]
        *
        df["UE_share"]
    )

    df["Commission_DD"] = (
        df["CommissionRate"]
        *
        df["DD_share"]
    )

    df["SD_Cost_Interaction"] = (
        df["DeliveryCostPerOrder"]
        *
        df["SD_share"]
    )

    df["Radius_SD"] = (
        df["DeliveryRadiusKM"]
        *
        df["SD_share"]
    )

    df["AdjustedDemand"] = (
        df["MonthlyOrders"]
        *
        df["GrowthFactor"]
    )

    return df