import pandas as pd
import plotly.express as px
from scipy import stats

from data.constants import TICKER_COL


def analysis(
    df_analysis: pd.DataFrame,
    feature: str = "rohrbach_exponential",
    target: str = "next_7d_log_returns",
    bin_feature: str = "trend_decile",
):
    df_analysis = df_analysis.dropna()

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_analysis[feature], df_analysis[target]
    )
    print(f"slope: {slope}")
    print(f"intercept: {intercept}")
    print(f"r_value: {r_value}")
    print(f"p_value: {p_value}")
    print(f"std_err: {std_err}")

    # Look at correlation between momentum signals and next week returns
    fig = px.scatter(
        df_analysis,
        x=feature,
        y=target,
        title=f"Crypto Relationship {feature} and {target}",
        trendline="ols",
        trendline_color_override="red",
        # facet_col=TICKER_COL,
        # facet_row="year",
        # facet_col="year",
        # facet_col_wrap=4,
    )
    # fig.update_xaxes(range=[-1.0, 1.0])
    # fig.update_yaxes(range=[-1.0, 1.0])
    # fig.update_xaxes(matches=None, showticklabels=True)
    # fig.update_yaxes(matches=None, showticklabels=True)
    fig.show()

    # Plot relationship between trend bins and next week returns
    df_tmp = (
        df_analysis.groupby(
            [
                bin_feature,
            ]
        )
        .agg({target: "mean"})
        .reset_index()
    )
    fig = px.bar(
        df_tmp,
        x=bin_feature,
        y=target,
    )
    fig.show()

    # Same analysis but by year
    df_tmp = (
        df_analysis.groupby(
            [
                "year",
                bin_feature,
            ]
        )
        .agg({target: "mean"})
        .reset_index()
    )
    fig = px.bar(
        df_tmp,
        x=bin_feature,
        y=target,
        facet_col="year",
    )
    fig.show()
