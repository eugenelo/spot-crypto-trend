import argparse
import pandas as pd
import plotly.express as px
from scipy import stats
import numpy as np
from datetime import datetime
import pytz
from pathlib import Path

from analysis import analysis
from backtest import (
    backtest_crypto,
)
from optimize import optimize_crypto
from simulation import simulation
from signal_generation import (
    create_analysis_signals,
    create_trading_signals,
)
from position_generation import (
    CRYPTO_MOMO_DEFAULT_PARAMS,
    generate_positions,
    generate_benchmark,
    nonempty_positions,
)
from utils import load_ohlc_to_daily_filtered


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
    parser.add_argument(
        "--initial_capital", "-c", type=int, help="Initial capital", default=12000
    )
    parser.add_argument("--skip_plots", action="store_true")
    # TODO(@eugene.lo): Support loading params
    return parser.parse_args()


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
    df_daily = load_ohlc_to_daily_filtered(
        args.input_path, input_freq=args.data_freq, tz=tz
    )

    # Create signals
    if args.mode == "analysis" or args.mode == "simulation":
        df_analysis = create_analysis_signals(df_daily)
    else:
        df_analysis = create_trading_signals(df_daily)

    # Validate dates
    data_start = df_analysis["timestamp"].min()
    if start_date < data_start:
        print(f"Input start_date is before start of data! Setting to {data_start}")
        start_date = data_start
    data_end = df_analysis["timestamp"].max()
    if end_date > data_end:
        print(f"Input end_date is after end of data! Setting to {data_end}")
        end_date = data_end

    if args.mode == "analysis":
        analysis(df_analysis)
    elif args.mode == "simulation":
        df_analysis = df_analysis.loc[
            (
                (df_analysis["timestamp"] >= start_date)
                & (df_analysis["timestamp"] <= end_date)
            )
        ]
        simulation(df_analysis)
    elif args.mode == "backtest":
        backtest_crypto(
            df_analysis,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.initial_capital,
            params=CRYPTO_MOMO_DEFAULT_PARAMS,
            skip_plots=args.skip_plots,
        )
    elif args.mode == "optimize":
        optimize_crypto(
            df_analysis,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.initial_capital,
            skip_subsample_plots=args.skip_plots,
        )
    elif args.mode == "positions":
        positions = generate_positions(df_analysis, CRYPTO_MOMO_DEFAULT_PARAMS)
        nonempty_positions = nonempty_positions(positions, timestamp=end_date)
        print(nonempty_positions)
        # Output to file
        if args.output_path is not None:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nonempty_positions[["timestamp", "ticker", "scaled_position"]].to_csv(
                str(output_path)
            )
            print(f"Wrote positions to '{output_path}'")
    else:
        raise ValueError("Unsupported mode")
