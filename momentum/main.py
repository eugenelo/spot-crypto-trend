import argparse
import logging
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yaml

from analysis.analysis import analysis
from core.constants import POSITION_COL, in_universe_excl_stablecoins
from core.utils import get_periods_per_day
from data.constants import DATETIME_COL, TICKER_COL
from data.utils import load_ohlc_to_daily_filtered  # noqa: F401
from data.utils import load_ohlc_to_hourly_filtered  # noqa: F401
from logging_custom.utils import setup_logging
from position_generation.benchmark import get_generate_benchmark_fn
from position_generation.constants import (
    SCALED_SIGNAL_COL,
    VOL_FORECAST_COL,
    VOL_TARGET_COL,
)
from position_generation.position_generation import get_generate_positions_fn
from position_generation.utils import nonempty_positions
from signal_generation.constants import get_signal_type
from signal_generation.signal_generation import (
    create_analysis_signals,
    create_trading_signals,
)
from simulation.backtest import backtest_crypto
from simulation.constants import (
    DEFAULT_REBALANCING_BUFFER,
    DEFAULT_REBALANCING_FREQ,
    DEFAULT_VOLUME_MAX_SIZE,
)
from simulation.optimize import optimize
from simulation.utils import rebal_freq_supported


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crypto Cross-Sectional Momentum + Trend Following"
    )
    parser.add_argument("mode", help="[analysis, backtest, optimize, positions]")
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
        "--initial_capital", "-c", type=float, help="Initial capital", default=12000
    )
    parser.add_argument("--skip_plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    pd.set_option("display.width", 2000)
    pd.set_option("display.precision", 2)

    # Configure logging
    log_config_path = Path(__file__).parent / Path(
        "../logging_custom/logging_config/momentum_config.yaml"
    )
    setup_logging(config_path=log_config_path)
    logger = logging.getLogger(__name__)

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
        ][DATETIME_COL]
    )

    # Load params
    params = {}
    if args.params_path is not None:
        with open(args.params_path, "r") as yaml_file:
            params = yaml.safe_load(yaml_file)
        logger.info(f"Loaded params: {params}")
    assert "signal" in params, "Signal should be specified in params!"
    # Lag positions if backtesting
    lag_positions = args.mode != "positions"
    generate_positions_fn = get_generate_positions_fn(
        params, periods_per_day=periods_per_day, lag_positions=lag_positions
    )

    # Create signals
    if args.mode == "analysis":
        df_analysis = create_analysis_signals(df_ohlc, periods_per_day=periods_per_day)
    else:
        df_analysis = create_trading_signals(
            df_ohlc,
            periods_per_day=periods_per_day,
            signal_type=get_signal_type(params),
        )

    # Validate dates
    data_start = df_analysis[DATETIME_COL].min()
    if start_date < data_start:
        logger.info(
            f"Input start_date is before start of data! Setting to {data_start}"
        )
        start_date = data_start
    data_end = df_analysis[DATETIME_COL].max()
    if end_date > data_end:
        logger.info(f"Input end_date is after end of data! Setting to {data_end}")
        end_date = data_end

    # Set input args
    rebalancing_freq = params.get("rebalancing_freq", DEFAULT_REBALANCING_FREQ)
    if rebalancing_freq is not None:
        assert rebal_freq_supported(rebalancing_freq), (
            f"Rebalancing frequency {rebalancing_freq} is not supported! Use a fixed"
            " frequency instead (e.g. days)."
        )
    volume_max_size = params.get("volume_max_size", DEFAULT_VOLUME_MAX_SIZE)
    rebalancing_buffer = params.get("rebalancing_buffer", DEFAULT_REBALANCING_BUFFER)
    logger.info(f"Rebalancing Freq: {rebalancing_freq}")
    logger.info(f"volume_max_size: {volume_max_size}")
    logger.info(f"rebalancing_buffer: {rebalancing_buffer}")

    if args.mode == "analysis":
        analysis(df_analysis)
    elif args.mode == "backtest":
        # Get position and benchmark generation functions
        assert (
            "generate_benchmark" in params
        ), "Benchmark generation function should be specified in params for backtest!"
        generate_benchmark_fn = get_generate_benchmark_fn(params)

        backtest_crypto(
            df_analysis,
            periods_per_day=periods_per_day,
            generate_positions_fn=generate_positions_fn,
            generate_benchmark_fn=generate_benchmark_fn,
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
        pd.set_option("display.precision", 4)
        pd.set_option(
            "display.float_format",
            partial(np.format_float_positional, precision=4, trim="0"),
        )

        positions = generate_positions_fn(df_analysis)
        df_positions = nonempty_positions(positions)
        df_positions = df_positions.loc[
            df_positions[DATETIME_COL] == df_positions[DATETIME_COL].max()
        ]

        # Add row containing column totals for printing only
        df_tmp = df_positions.copy().sort_values(by=POSITION_COL, ascending=False)
        df_tmp.loc["total"] = df_tmp.sum(numeric_only=True, axis=0)

        cols_of_interest = [
            DATETIME_COL,
            TICKER_COL,
            VOL_FORECAST_COL,
            VOL_TARGET_COL,
            SCALED_SIGNAL_COL,
            POSITION_COL,
        ]
        logger.info(df_tmp[cols_of_interest])
        # Output to file
        if args.output_path is not None:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_positions[cols_of_interest].to_csv(str(output_path), index=False)
            logger.info(f"Wrote positions to '{output_path}'")
    else:
        raise ValueError("Unsupported mode")
