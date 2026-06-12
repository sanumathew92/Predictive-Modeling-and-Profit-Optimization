def generate_recommendations(
        uplift_pct,
        optimal_sd,
        optimal_commission,
        probability_of_loss,
        growth_factor):

    recommendations = []

    if uplift_pct > 10:

        recommendations.append(
            "Significant optimization opportunity exists."
        )

    if optimal_sd > 0.5:

        recommendations.append(
            "Increase investment in self-delivery infrastructure."
        )

    if optimal_commission < 0.20:

        recommendations.append(
            "Negotiate aggregator commissions."
        )

    if probability_of_loss < 5:

        recommendations.append(
            "Business risk profile is low."
        )

    if growth_factor > 1.03:

        recommendations.append(
            "Strong growth opportunity exists."
        )

    return recommendations