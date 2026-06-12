import pandas as pd


def calculate_sensitivity(
        base_input,
        model,
        variables,
        create_features):

    current_profit = model.predict(base_input)[0]

    sensitivity = {}

    for var in variables:

        high = base_input.copy()

        low = base_input.copy()

        high[var] *= 1.1

        low[var] *= 0.9

        high = create_features(high)

        low = create_features(low)

        p_high = model.predict(high)[0]

        p_low = model.predict(low)[0]

        sensitivity[var] = (
            p_high - p_low
        )

    sensitivity_df = pd.DataFrame({

        "Variable": sensitivity.keys(),

        "Impact": sensitivity.values()

    })

    sensitivity_df = sensitivity_df.sort_values(
        "Impact"
    )

    return sensitivity_df