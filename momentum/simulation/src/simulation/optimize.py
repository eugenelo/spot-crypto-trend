import vectorbt as vbt
import pandas as pd
import plotly.express as px
import numpy as np
from dataclasses import dataclass
from functools import reduce
import datetime
from dateutil.relativedelta import relativedelta
from tqdm.auto import tqdm
from typing import List, Callable, Optional

from simulation.backtest import backtest
from simulation.stats import get_stats_of_interest, plot_cumulative_returns
from position_generation.v1 import (
    CRYPTO_MOMO_DEFAULT_PARAMS,
    generate_positions_v1,
)
from position_generation.benchmark import (
    generate_benchmark_btc,
)


@dataclass
class OptimizationResult:
    id: int
    portfolio: vbt.Portfolio
    start_date: datetime.datetime
    end_date: datetime.datetime
    params: dict
    max_total_num_positions_long: int


def generate_subsample_periods(start_date, end_date, dt: relativedelta):
    # Subsample non-overlapping x month periods
    subsample_periods = []
    subsample_start = start_date
    subsample_end = start_date + dt
    while subsample_end < end_date:
        subsample_periods.append(
            (subsample_start, subsample_end + relativedelta(days=-1))
        )
        subsample_start += dt
        subsample_end += dt
    # Add the last period if not empty
    if (subsample_end - subsample_start).days > 0:
        subsample_periods.append(
            (subsample_start, subsample_end + relativedelta(days=-1))
        )
    # Remove any periods which are too short
    return subsample_periods


def optimize(
    df_analysis,
    periods_per_day,
    initial_capital,
    start_date,
    end_date,
    subsample_dt=None,
    fixed_params=None,
) -> List[OptimizationResult]:
    DEBUG = False

    # Set param sweeps
    if fixed_params is not None:
        momentum_factors = [fixed_params["momentum_factor"]]
        position_types = [fixed_params["type"]]
        num_asset_thresholds = [fixed_params["num_assets_to_keep"]]
        min_signal_thresholds = [fixed_params["min_signal_threshold"]]
        signal_spread = [
            fixed_params["max_signal_threshold"] - fixed_params["min_signal_threshold"]
        ]
    elif DEBUG:
        momentum_factors = ["30d_log_returns"]
        position_types = ["simple"]
        num_asset_thresholds = [200]
        min_signal_thresholds = [0.00, 0.02]
        signal_spread = [0.01, 0.08]
    else:
        momentum_factors = [
            "15d_log_returns",
            "21d_log_returns",
            "30d_log_returns",
            # "42d_log_returns",
        ]
        position_types = [
            "simple",
            # "decile",
            # "crossover",
        ]
        num_asset_thresholds = [int(1e6)]
        min_signal_thresholds = [0.00, 0.02, 0.05, 0.1, 0.15]
        signal_spread = [0.05, 0.1, 0.2]
    # Set rebalancing sweep
    if subsample_dt is not None:
        rebalancing_freqs = [None, "1d"]
    elif DEBUG:
        rebalancing_freqs = [None, "1d", "7d"]
    else:
        rebalancing_freqs = [None, "1d", "2d", "3d", "4d", "7d", "10d", "14d"]

    # Subsample non-overlapping 4 month periods
    subsample_periods = [(start_date, end_date)]
    if subsample_dt is not None:
        subsample_periods += generate_subsample_periods(
            start_date, end_date, dt=subsample_dt
        )

    total_iter = (
        len(momentum_factors)
        * len(position_types)
        * len(num_asset_thresholds)
        * len(min_signal_thresholds)
        * len(signal_spread)
        * len(rebalancing_freqs)
        * len(subsample_periods)
    )
    out = []
    id = 0
    pbar = tqdm(desc="Hyperparameter Optimization", total=total_iter)
    for momentum_factor in momentum_factors:
        for position_type in position_types:
            for num_assets_to_keep in num_asset_thresholds:
                for min_signal_threshold in min_signal_thresholds:
                    for spread in signal_spread:
                        # Generate positions from params
                        max_signal_threshold = min_signal_threshold + spread
                        pf_params = CRYPTO_MOMO_DEFAULT_PARAMS.copy()
                        pf_params["momentum_factor"] = momentum_factor
                        pf_params["num_assets_to_keep"] = num_assets_to_keep
                        pf_params["min_signal_threshold"] = min_signal_threshold
                        pf_params["max_signal_threshold"] = max_signal_threshold
                        pf_params["type"] = position_type
                        pf_positions = generate_positions_v1(df_analysis, pf_params)
                        for rebalancing_freq in rebalancing_freqs:
                            # Rebalancing takes place during the actual backtest by resampling, no need to regenerate positions
                            pf_params["rebalancing_freq"] = rebalancing_freq
                            pf_params["id"] = id
                            id += 1
                            for subsample_period in subsample_periods:
                                (
                                    pf_params["start_date"],
                                    pf_params["end_date"],
                                ) = subsample_period
                                # print(pf_params)

                                # Run backtest
                                pf = backtest(
                                    pf_positions,
                                    periods_per_day=periods_per_day,
                                    initial_capital=initial_capital,
                                    rebalancing_freq=pf_params["rebalancing_freq"],
                                    start_date=pf_params["start_date"],
                                    end_date=pf_params["end_date"],
                                    verbose=False,
                                )
                                out.append(
                                    OptimizationResult(
                                        id=pf_params["id"],
                                        portfolio=pf,
                                        start_date=pf_params["start_date"],
                                        end_date=pf_params["end_date"],
                                        params=pf_params.copy(),
                                        max_total_num_positions_long=pf_positions[
                                            "total_num_positions_long"
                                        ].max(),
                                    )
                                )
                                pbar.update(1)
    return out


def optimize_crypto(
    df_analysis: pd.DataFrame,
    periods_per_day: int,
    start_date: datetime,
    end_date: datetime,
    initial_capital: int,
    skip_subsample_plots: bool = False,
) -> OptimizationResult:
    # Optimize the parameters related to signal generation
    subsample_dt = relativedelta(months=+4)
    opt_res_signal_gen = optimize(
        df_analysis,
        periods_per_day=periods_per_day,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        subsample_dt=subsample_dt,
    )

    # Average the stats for the subsample periods belonging to the same set of parameters, compare with stats over the total period
    full_period_results = [
        x
        for x in opt_res_signal_gen
        if x.start_date == start_date and x.end_date == end_date
    ]
    subsample_results = [
        x
        for x in opt_res_signal_gen
        if x.start_date != start_date or x.end_date != end_date
    ]
    unique_ids = [x.id for x in full_period_results]

    full_period_stats = [
        get_stats_of_interest(x.portfolio, name=x.id)
        for x in tqdm(full_period_results, desc="Full Period Results")
    ]
    pbar = tqdm(desc="Subsample Results", total=len(unique_ids))
    subsample_results_averaged_stats = []
    for id in unique_ids:
        subsample_results_for_id = [x for x in subsample_results if x.id == id]
        subsample_stats = [
            get_stats_of_interest(x.portfolio, name=i)
            for i, x in enumerate(subsample_results_for_id)
        ]
        df_stats = reduce(
            lambda left, right: pd.merge(left, right, on=["index"], how="outer"),
            subsample_stats,
        ).set_index("index")
        df_stats = df_stats.T
        df_mean = df_stats.mean().T.reset_index()
        df_mean.rename(columns={0: id}, inplace=True)
        subsample_results_averaged_stats.append(df_mean)
        pbar.update(1)

    # Full period stats
    df_stats_full_period = reduce(
        lambda left, right: pd.merge(left, right, on=["index"], how="outer"),
        full_period_stats,
    ).set_index("index")
    df_stats_full_period = df_stats_full_period.T

    # Subsample average stats
    df_stats_subsample_average = reduce(
        lambda left, right: pd.merge(left, right, on=["index"], how="outer"),
        subsample_results_averaged_stats,
    ).set_index("index")
    df_stats_subsample_average = df_stats_subsample_average.T

    # Sanity check sufficient overlap between full and subsample
    top = max(int(0.1 * len(df_stats_full_period)), 20)
    comparison_metric = "Total Return [%]"
    best_portfolios_full_idx = (
        df_stats_full_period.sort_values(by=[comparison_metric], ascending=[False])
        .head(top)
        .index
    )
    best_portfolios_subsample_idx = (
        df_stats_subsample_average.sort_values(
            by=[comparison_metric], ascending=[False]
        )
        .head(top)
        .index
    )
    common_idx = best_portfolios_full_idx.intersection(best_portfolios_subsample_idx)
    print(
        f"Overlap between top max(10%, 20) full vs subsample: {len(common_idx) / top * 100}%"
    )

    # Choose the most stable results across subsamples, analyze the stats over the full period
    top = 10
    best_portfolios_subsample_idx = best_portfolios_subsample_idx[:top]

    print("Params")
    for idx in best_portfolios_subsample_idx:
        print(f"{idx}: {full_period_results[idx].params}")
    print()
    print("Subsample Averages")
    print(
        df_stats_subsample_average.iloc[best_portfolios_subsample_idx][
            [
                "Start Value",
                "End Value",
                "Total Return [%]",
                "Total Fees Paid",
                "Max Drawdown [%]",
                "Sharpe Ratio",
                "Annualized Return [%]",
                "Annualized Volatility [%]",
            ]
        ].head(top)
    )
    print()
    print("Full")
    print(
        df_stats_full_period.iloc[best_portfolios_subsample_idx][
            [
                "Start Value",
                "End Value",
                "Total Return [%]",
                "Total Fees Paid",
                "Max Drawdown [%]",
                "Sharpe Ratio",
                "Annualized Return [%]",
                "Annualized Volatility [%]",
            ]
        ].head(top)
    )
    print()

    # Benchmark is 100% BTC
    df_benchmark = generate_benchmark_btc(df_analysis)
    pf_benchmark = backtest(
        df_benchmark,
        periods_per_day=periods_per_day,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        verbose=False,
    )

    if not skip_subsample_plots:
        # Plot cumulative returns for best portfolios + benchmark on sub time periods
        subsample_periods = generate_subsample_periods(
            start_date, end_date, dt=subsample_dt
        )
        for period in subsample_periods:
            subsample_start, subsample_end = period
            subsample_portfolios_for_period = []
            subsample_portfolio_names = []
            for res in subsample_results:
                if (
                    res.start_date == subsample_start
                    and res.end_date == subsample_end
                    and res.id in best_portfolios_subsample_idx
                ):
                    subsample_portfolios_for_period.append(res.portfolio)
                    subsample_portfolio_names.append(res.id)
            assert len(subsample_portfolio_names) == len(best_portfolios_subsample_idx)
            # Re-sort to make consistent with the full time period plot
            sorting_order = []
            for idx in best_portfolios_subsample_idx:
                for i, name in enumerate(subsample_portfolio_names):
                    if name == idx:
                        sorting_order.append(i)
                        break
            assert len(sorting_order) == len(subsample_portfolio_names)
            subsample_portfolios_for_period = [
                subsample_portfolios_for_period[i] for i in sorting_order
            ]
            subsample_portfolio_names = best_portfolios_subsample_idx
            plot_cumulative_returns(
                subsample_portfolios_for_period, subsample_portfolio_names
            )

    # Plot cumulative returns for best portfolios + benchmark on full time period
    best_portfolios = [
        full_period_results[idx].portfolio for idx in best_portfolios_subsample_idx
    ] + [pf_benchmark]
    best_portfolio_names = [
        full_period_results[idx].id for idx in best_portfolios_subsample_idx
    ] + ["Benchmark"]
    plot_cumulative_returns(best_portfolios, best_portfolio_names)

    # Choose the desired parameter set
    chosen_idx = input("Enter desired parameter set id:")
    best_params = full_period_results[int(chosen_idx)].params.copy()
    del best_params["rebalancing_freq"]
    print(f"Params: {best_params}")

    # Optimize over rebalancing frequency
    opt_res_rebalancing = optimize(
        df_analysis,
        periods_per_day=periods_per_day,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        subsample_dt=None,
        fixed_params=best_params,
    )
    full_period_stats_rebal = [
        get_stats_of_interest(x.portfolio, name=x.id)
        for x in tqdm(opt_res_rebalancing, desc="Full Period Results - Rebal")
    ]
    # Full period stats
    df_stats_full_period_rebal = reduce(
        lambda left, right: pd.merge(left, right, on=["index"], how="outer"),
        full_period_stats_rebal,
    ).set_index("index")
    df_stats_full_period_rebal = df_stats_full_period_rebal.T

    # Full period stats
    print(
        df_stats_full_period_rebal.sort_values(
            by=[comparison_metric], ascending=[False]
        )[
            [
                "Start Value",
                "End Value",
                "Total Return [%]",
                "Total Fees Paid",
                "Max Drawdown [%]",
                "Sharpe Ratio",
                "Annualized Return [%]",
                "Annualized Volatility [%]",
            ]
        ]
    )
    print()

    # Plot cumulative returns for best portfolios + benchmark
    best_portfolios_rebal_idx = df_stats_full_period_rebal.sort_values(
        by=[comparison_metric], ascending=[False]
    ).index
    best_portfolios = [
        opt_res_rebalancing[idx].portfolio for idx in best_portfolios_rebal_idx[:top]
    ] + [pf_benchmark]
    best_portfolio_names = [
        opt_res_rebalancing[idx].params["rebalancing_freq"]
        for idx in best_portfolios_rebal_idx[:top]
    ] + ["Benchmark"]
    plot_cumulative_returns(best_portfolios, best_portfolio_names)

    best_rebal_idx = best_portfolios_rebal_idx[0]
    print(f"Final Params: {opt_res_rebalancing[best_rebal_idx].params}")
    return opt_res_rebalancing[best_rebal_idx]


def optimize_rebalancing_buffer(
    df_analysis: pd.DataFrame,
    periods_per_day: int,
    generate_positions: Callable,
    start_date: datetime,
    end_date: datetime,
    initial_capital: int,
    rebalancing_freq: Optional[str],
    volume_max_size: float,
    vol_target: Optional[float],
    skip_plots: bool,
):
    # Generate positions
    df_portfolio = generate_positions(df_analysis)

    # Optimize value of rebalancing buffer which yields highest net sharpe
    rebal_buffer_to_pf = {}
    step_size = 0.0005
    num_steps = 200
    for i in range(num_steps):
        rebalancing_buffer = i * step_size
        pf_portfolio = backtest(
            df_portfolio,
            periods_per_day=periods_per_day,
            initial_capital=initial_capital,
            rebalancing_freq=rebalancing_freq,
            start_date=start_date,
            end_date=end_date,
            with_fees=True,
            volume_max_size=volume_max_size,
            rebalancing_buffer=rebalancing_buffer,
            verbose=False,
        )
        rebal_buffer_to_pf[rebalancing_buffer] = pf_portfolio

    # Get stats per portfolio
    stats = []
    portfolios = []
    pf_names = []
    for rebalancing_buffer, pf in rebal_buffer_to_pf.items():
        if vol_target is not None:
            # Disqualify buffers which are too large, exceeding volatility target
            realized_vol = pf.annualized_volatility()
            if abs(realized_vol - vol_target) / vol_target > 0.25:
                continue
        pf_name = f"Rebal Buffer {rebalancing_buffer * 100:.4f}%"
        pf_stats = get_stats_of_interest(pf, name=pf_name)
        stats.append(pf_stats)
        portfolios.append(pf)
        pf_names.append(pf_name)
    df_stats = reduce(
        lambda left, right: pd.merge(left, right, on=["index"], how="outer"),
        stats,
    ).set_index("index")
    df_stats = df_stats.T

    comparison_metric = "Sharpe Ratio"
    print(
        df_stats.sort_values(by=[comparison_metric], ascending=[False])[
            [
                "Sharpe Ratio",
                "Total Fees Paid [$]",
                "Total Trades",
                "Total Return [%]",
                "Max Drawdown [%]",
                "Annualized Return [%]",
                "Annualized Volatility [%]",
                "Avg Daily Turnover [%]",
            ]
        ].head(25)
    )
    print()

    if not skip_plots:
        # best_portfolios_idx = df_stats.sort_values(
        #     by=[comparison_metric], ascending=[False]
        # ).index
        plot_cumulative_returns(portfolios, pf_names)
