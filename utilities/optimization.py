from scipy.optimize import minimize


def optimize_channel_mix(
        base_input,
        model,
        create_features,
        feature_names):

    def objective(x):

        ue = x[0]

        dd = x[1]

        sd = x[2]

        temp = base_input.copy()

        temp["UE_share"] = ue

        temp["DD_share"] = dd

        temp["SD_share"] = sd

        temp = create_features(temp)

        temp = temp[feature_names]

        profit = model.predict(temp)[0]

        return -profit


    constraint = {

        'type':'eq',

        'fun':lambda x:

        x[0]+x[1]+x[2]-1

    }

    bounds = [

        (0,1),

        (0,1),

        (0,1)

    ]

    x0 = [

        base_input["UE_share"][0],

        base_input["DD_share"][0],

        base_input["SD_share"][0]

    ]

    result = minimize(

        objective,

        x0,

        bounds=bounds,

        constraints=constraint

    )

    return result