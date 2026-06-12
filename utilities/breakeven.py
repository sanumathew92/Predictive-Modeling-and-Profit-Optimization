import numpy as np


def break_even_commission(
        base_input,
        model,
        create_features,
        feature_names):

    commissions = np.arange(
        0.05,
        0.40,
        0.01
    )

    profits = []

    for c in commissions:

        temp = base_input.copy()

        temp["CommissionRate"] = c

        temp = create_features(temp)

        temp = temp[feature_names]

        p = model.predict(temp)[0]

        profits.append(p)

    return commissions, profits