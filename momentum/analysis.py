import pandas as pd
import plotly.express as px
from scipy import stats


def analysis(df_analysis: pd.DataFrame):
    # Linear regression
    target = "30d_log_returns"
    feature = "next_6d_log_returns"
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_analysis[target], df_analysis[feature]
    )
    print(f"slope: {slope}")
    print(f"intercept: {intercept}")
    print(f"r_value: {r_value}")
    print(f"p_value: {p_value}")
    print(f"std_err: {std_err}")

    # Look at correlation between momentum signals and next 6-day returns
    fig = px.scatter(
        df_analysis,
        x=target,
        y=feature,
        title=f"Crypto Relationship {target.split('_')[0]} Return and Next {feature.split('_')[1]} Return",
        trendline="ols",
        trendline_color_override="red",
        # facet_col="ticker",
        # facet_row="year",
        # facet_col="year",
        # facet_col_wrap=4,
    )
    fig.update_xaxes(range=[-1.0, 1.0])
    fig.update_yaxes(range=[-1.0, 1.0])
    # fig.update_xaxes(matches=None, showticklabels=True)
    # fig.update_yaxes(matches=None, showticklabels=True)
    fig.show()

    # Plot relationship between trend bins and next week returns
    df_tmp = (
        df_analysis.groupby(
            [
                # "ticker",
                "30d_log_quintile",
            ]
        )
        .agg({"next_6d_log_returns": "mean"})
        .reset_index()
    )
    fig = px.bar(
        df_tmp,
        x="30d_log_quintile",
        y="next_6d_log_returns",
        # facet_col="ticker",
    )
    fig.show()

    print(
        df_analysis.sort_values(by=["next_6d_returns"], ascending=False)[
            [
                "ticker",
                "timestamp",
                "close",
                "dollar_volume",
                "30d_returns",
                "next_6d_returns",
            ]
        ].head(20)
    )
    # print(
    #     df_analysis.loc[
    #         (df_analysis["ticker"] == "USTUSD")
    #         & (df_analysis["year"] == 2022)
    #         & (df_analysis["month"] == 6)
    #     ][["ticker", "timestamp", "close", "30d_returns", "next_7d_returns"]]
    # )
