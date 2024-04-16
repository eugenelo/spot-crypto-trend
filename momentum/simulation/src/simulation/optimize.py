import datetime
from dataclasses import dataclass
from functools import reduce
from itertools import product
from typing import List, Optional

import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm.auto import tqdm

from position_generation.benchmark import generate_benchmark_btc  # noqa: F401
from position_generation.position_generation import generate_positions
from position_generation.utils import Direction
from simulation.backtest import backtest
from simulation.stats import get_stats_of_interest, plot_cumulative_returns
from simulation.vbt import get_annualized_volatility, vbt


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


@dataclass(eq=False)
class OptimizeParameterSet:
    name: str
    signal: str
    direction: Direction
    rebalancing_freq: Optional[float]
    volatility_target: Optional[float]
    rebalancing_buffer: Optional[float]
    cross_sectional_percentage: Optional[float]
    cross_sectional_equal_weight: Optional[bool]
    min_daily_volume: Optional[float]
    max_daily_volume: Optional[float]
    with_fees: bool
    leverage: float


def generate_parameter_sets(optimize_params: dict) -> List[OptimizeParameterSet]:
    # Generate all combinations of parameter set
    param_names = [
        "signal",
        "direction",
        "rebalancing_freq",
        "volatility_target",
        "rebalancing_buffer",
        "cross_sectional_percentage",
        "cross_sectional_equal_weight",
        "min_daily_volume",
        "max_daily_volume",
        "with_fees",
        "leverage",
    ]
    default_values = {
        "rebalancing_freq": None,
        "volatility_target": None,
        "rebalancing_buffer": None,
        "cross_sectional_percentage": None,
        "cross_sectional_equal_weight": False,
        "min_daily_volume": None,
        "max_daily_volume": None,
        "with_fees": True,
        "leverage": 1.0,
    }
    params = []
    for param_name in param_names:
        if param_name in optimize_params:
            param = optimize_params[param_name]
        elif param_name in default_values:
            param = default_values[param_name]
        else:
            raise RuntimeError(f"{param_name} must be specified!")

        # Correct type
        if param_name == "direction":
            if type(param) is list:
                param = [Direction(x) for x in param]
            else:
                param = Direction(param)

        # Append to list
        if type(param) is list:
            params.append(param)
        elif type(param) is dict:
            # Parse dict for step size & num steps
            assert (
                "step_size" in param and "num_steps" in param
            ), f"{param_name} - Invalid param definition!"
            params.append(
                [param["step_size"] * i for i in range(1, param["num_steps"] + 1)]
            )
        else:
            params.append([param])

    param_idx_of_interest = []
    for idx, param in enumerate(params):
        if len(param) > 1:
            # Actually being optimized over, include in name
            param_idx_of_interest.append(idx)

    out = []
    for param_set in product(*params):
        param_strs = []
        for idx in param_idx_of_interest:
            param = param_set[idx]
            if isinstance(param, float):
                param_str = f"{param:.4g}"
            else:
                param_str = str(param)
            param_strs.append(f"{param_names[idx]}: {param_str}")
        pf_name = ", ".join(param_strs)
        out.append(OptimizeParameterSet(pf_name, *param_set))
    return out


def optimize(
    df_analysis: pd.DataFrame,
    periods_per_day: int,
    optimize_params: dict,
    start_date: datetime,
    end_date: datetime,
    initial_capital: int,
    volume_max_size: float,
    skip_plots: bool,
) -> List[OptimizationResult]:
    # General params (not position generation related)
    parameter_sets = generate_parameter_sets(optimize_params)

    # Optimize value of rebalancing buffer which yields highest net sharpe
    params_to_pf = {}
    for param_set in tqdm(parameter_sets):
        df_portfolio = generate_positions(
            df_analysis,
            signal=param_set.signal,
            periods_per_day=periods_per_day,
            direction=param_set.direction,
            volatility_target=param_set.volatility_target,
            cross_sectional_percentage=param_set.cross_sectional_percentage,
            cross_sectional_equal_weight=param_set.cross_sectional_equal_weight,
            min_daily_volume=param_set.min_daily_volume,
            max_daily_volume=param_set.max_daily_volume,
            leverage=param_set.leverage,
            lag_positions=True,
        )
        pf_portfolio = backtest(
            df_portfolio,
            periods_per_day=periods_per_day,
            initial_capital=initial_capital,
            leverage=param_set.leverage,
            rebalancing_freq=param_set.rebalancing_freq,
            start_date=start_date,
            end_date=end_date,
            with_fees=param_set.with_fees,
            volume_max_size=volume_max_size,
            rebalancing_buffer=param_set.rebalancing_buffer,
            verbose=False,
        )
        params_to_pf[param_set] = pf_portfolio

    # Get stats per portfolio
    stats = []
    portfolios = []
    pf_names = []
    for param_set, pf in params_to_pf.items():
        if param_set.volatility_target is not None and param_set.volatility_target > 0:
            # Disqualify buffers which are too large, exceeding volatility target
            realized_vol = get_annualized_volatility(pf)
            if (
                abs(realized_vol - param_set.volatility_target)
                / param_set.volatility_target
                > 0.25
            ):
                continue
        pf_name = param_set.name
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
    top = 25
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
                "Max Gross Exposure [%]",
            ]
        ].head(top)
    )
    print()

    if not skip_plots:
        best_portfolios_idx = (
            df_stats.reset_index()
            .sort_values(by=[comparison_metric], ascending=[False])
            .head(top)
            .index
        )
        pf_plot = [portfolios[i] for i in best_portfolios_idx]
        pf_names_plot = [pf_names[i] for i in best_portfolios_idx]
        plot_cumulative_returns(pf_plot, pf_names_plot)
