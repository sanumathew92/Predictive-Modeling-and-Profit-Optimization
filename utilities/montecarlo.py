import numpy as np


def run_monte_carlo(
        base_input,
        model,
        create_features,
        feature_names,
        n_simulations=5000):

    profits = []

    for i in range(n_simulations):

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

        temp = create_features(temp)

        temp = temp[feature_names]

        profit = model.predict(temp)[0]

        profits.append(profit)

    return np.array(profits)