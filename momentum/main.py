import argparse
import pandas as pd
import plotly.express as px
from scipy import stats
import numpy as np
from datetime import datetime
import pytz
from pathlib import Path
import yaml
from typing import Callable

from analysis.analysis import analysis
from simulation.simulation import simulation
from simulation.utils import rebal_freq_supported
from simulation.backtest import backtest_crypto
from simulation.optimize import optimize
from signal_generation.signal_generation import (
    create_analysis_signals,
    create_trading_signals,
)
from signal_generation.constants import SignalType
from simulation.constants import DEFAULT_VOLUME_MAX_SIZE, DEFAULT_REBALANCING_BUFFER
from position_generation.benchmark import generate_benchmark_btc
from position_generation.utils import nonempty_positions, Direction
from position_generation.v1 import generate_positions_v1
from position_generation.generate_positions import generate_positions
from data.utils import load_ohlc_to_hourly_filtered, load_ohlc_to_daily_filtered
from core.utils import get_periods_per_day
from core.constants import (
    TIMESTAMP_COL,
    TICKER_COL,
    POSITION_COL,
    in_universe_excl_stablecoins,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crypto Cross-Sectional Momentum + Trend Following"
    )
    parser.add_argument(
        "mode", help="[analysis, simulation, backtest, optimize, positions]"
    )
    parser.add_argument(
        "--input_path", "-i", type=str, required=True, help="Input file path"
    )
    parser.add_argument("--output_path", "-o", type=str, help="Output file path")
    parser.add_argument(
        "--data_freq", "-f", type=str, help="Data frequency", required=True
    )
    parser.add_argument(
        "--start_date",
        "-s",
        type=str,
        help="Start date inclusive",
        default="1900-01-01",
    )
    parser.add_argument(
        "--end_date", "-e", type=str, help="End date inclusive", default="2100-01-01"
    )
    parser.add_argument("--timezone", "-t", type=str, help="Timezone", default="UTC")
    parser.add_argument("--params_path", "-p", type=str, help="Params yaml file path")
    parser.add_argument(
        "--rebalancing_freq", "-r", type=str, help="Rebalancing frequency"
    )
    parser.add_argument(
        "--initial_capital", "-c", type=float, help="Initial capital", default=12000
    )
    parser.add_argument("--skip_plots", action="store_true")
    return parser.parse_args()


def get_signal_type(params: dict) -> SignalType:
    if params["signal"] == "v1":
        return SignalType.HistoricalReturns
    elif params["signal"].startswith("rohrbach"):
        return SignalType.Rohrbach
    raise ValueError(
        f"Unsupported 'generate_positions' argument: {params['generate_positions']}"
    )


def get_generate_positions(params: dict, periods_per_day: int) -> Callable:
    if params["signal"] == "v1":
        v1_params = params["params"]
        generate_positions_fn = lambda df: generate_positions_v1(df, params=v1_params)
    elif params["signal"].startswith("rohrbach"):
        signal = params["signal"]
        direction = Direction(params["direction"])
        volatility_target = params.get("volatility_target", None)
        cross_sectional_percentage = params.get("cross_sectional_percentage", None)
        cross_sectional_equal_weight = params.get("cross_sectional_equal_weight", False)
        min_daily_volume = params.get("min_daily_volume", None)
        max_daily_volume = params.get("max_daily_volume", None)
        leverage = params.get("leverage", 1.0)
        generate_positions_fn = lambda df: generate_positions(
            df,
            signal=signal,
            periods_per_day=periods_per_day,
            direction=direction,
            volatility_target=volatility_target,
            cross_sectional_percentage=cross_sectional_percentage,
            cross_sectional_equal_weight=cross_sectional_equal_weight,
            min_daily_volume=min_daily_volume,
            max_daily_volume=max_daily_volume,
            leverage=leverage,
        )
    else:
        raise ValueError(
            f"Unsupported 'generate_positions' argument: {params['generate_positions']}"
        )
    return generate_positions_fn


def get_generate_benchmark(params: dict) -> Callable:
    if params["generate_benchmark"] == "btc":
        generate_benchmark = generate_benchmark_btc
    else:
        raise ValueError(
            f"Unsupported 'generate_benchmark' argument: {params['generate_benchmark']}"
        )
    return generate_benchmark


if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    pd.set_option("display.width", 2000)
    pd.set_option("display.precision", 2)

    # Parse arguments
    args = parse_args()
    tz = pytz.timezone(args.timezone)
    start_date = tz.localize(
        datetime.strptime(args.start_date.replace("/", "-"), "%Y-%m-%d")
    )
    end_date = tz.localize(
        datetime.strptime(args.end_date.replace("/", "-"), "%Y-%m-%d")
    )

    # Parse data
    df_ohlc = load_ohlc_to_daily_filtered(
        args.input_path,
        input_freq=args.data_freq,
        tz=tz,
        whitelist_fn=in_universe_excl_stablecoins,
    )
    # Infer periods per day from the timestamp column for the first ticker
    periods_per_day = get_periods_per_day(
        timestamp_series=df_ohlc.loc[
            df_ohlc[TICKER_COL] == df_ohlc[TICKER_COL].unique()[0]
        ][TIMESTAMP_COL]
    )

    # Load params
    params = {}
    if args.params_path is not None:
        with open(args.params_path, "r") as yaml_file:
            params = yaml.safe_load(yaml_file)
        print(f"Loaded params: {params}")
    assert "signal" in params, "Signal should be specified in params!"
    generate_positions_fn = get_generate_positions(params, periods_per_day)

    # Create signals
    if args.mode == "analysis" or args.mode == "simulation":
        df_analysis = create_analysis_signals(df_ohlc, periods_per_day=periods_per_day)
    else:
        df_analysis = create_trading_signals(
            df_ohlc,
            periods_per_day=periods_per_day,
            signal_type=get_signal_type(params),
        )

    # Validate dates
    data_start = df_analysis[TIMESTAMP_COL].min()
    if start_date < data_start:
        print(f"Input start_date is before start of data! Setting to {data_start}")
        start_date = data_start
    data_end = df_analysis[TIMESTAMP_COL].max()
    if end_date > data_end:
        print(f"Input end_date is after end of data! Setting to {data_end}")
        end_date = data_end

    # Set input args
    rebalancing_freq = args.rebalancing_freq
    if "rebalancing_freq" in params:
        if rebalancing_freq is None:
            rebalancing_freq = params["rebalancing_freq"]
        elif params["rebalancing_freq"] != rebalancing_freq:
            print(
                f"Rebalancing freq conflict! Params={params['rebalancing_freq']}, Input={rebalancing_freq}. Using input {rebalancing_freq}."
            )
    if rebalancing_freq is not None:
        assert rebal_freq_supported(
            rebalancing_freq
        ), f"Rebalancing frequency {rebalancing_freq} is not supported! Use a fixed frequency instead (e.g. days)."
    print(f"Rebalancing Freq: {rebalancing_freq}")
    volume_max_size = DEFAULT_VOLUME_MAX_SIZE
    if "volume_max_size" in params:
        volume_max_size = params["volume_max_size"]
    print(f"volume_max_size: {volume_max_size}")
    rebalancing_buffer = DEFAULT_REBALANCING_BUFFER
    if "rebalancing_buffer" in params:
        rebalancing_buffer = params["rebalancing_buffer"]
    print(f"rebalancing_buffer: {rebalancing_buffer}")

    if args.mode == "analysis":
        analysis(df_analysis)
    elif args.mode == "simulation":
        df_analysis = df_analysis.loc[
            (
                (df_analysis[TIMESTAMP_COL] >= start_date)
                & (df_analysis[TIMESTAMP_COL] <= end_date)
            )
        ]
        simulation(df_analysis)
    elif args.mode == "backtest":
        # Get position and benchmark generation functions
        assert (
            "generate_benchmark" in params
        ), "Benchmark generation function should be specified in params for backtest!"
        generate_benchmark_fn = get_generate_benchmark(params)

        backtest_crypto(
            df_analysis,
            periods_per_day=periods_per_day,
            generate_positions=generate_positions_fn,
            generate_benchmark=generate_benchmark_fn,
            start_date=start_date,
            end_date=end_date,
            rebalancing_freq=rebalancing_freq,
            initial_capital=args.initial_capital,
            leverage=params.get("leverage", 1.0),
            volume_max_size=volume_max_size,
            rebalancing_buffer=rebalancing_buffer,
            skip_plots=args.skip_plots,
        )
    elif args.mode == "optimize":
        optimize(
            df_analysis,
            periods_per_day=periods_per_day,
            optimize_params=params,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.initial_capital,
            volume_max_size=volume_max_size,
            skip_plots=args.skip_plots,
        )
    elif args.mode == "positions":
        positions = generate_positions_fn(df_analysis)
        nonempty_positions = nonempty_positions(positions, timestamp=end_date)
        print(nonempty_positions)
        # Output to file
        if args.output_path is not None:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nonempty_positions[[TIMESTAMP_COL, TICKER_COL, POSITION_COL]].to_csv(
                str(output_path)
            )
            print(f"Wrote positions to '{output_path}'")
    else:
        raise ValueError("Unsupported mode")
