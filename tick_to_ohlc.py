import argparse
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Convert tick data to OHLC data")
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input file path"
    )
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument(
        "--timeframe", "-t", type=str, default="1H", help="1Min / 1H / 1D / etc."
    )
    parser.add_argument(
        "--start",
        "-s",
        type=str,
        help="Starting UTC date before which to filter out data (inclusive).",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=str,
        help="Ending UTC date after which to filter out data (inclusive).",
    )
    return parser.parse_args()


def tick_to_ohlc(input: Path, timeframe: str, start: Optional[str], end: Optional[str]):
    """
    Convert tick data to OHLC data.

    Parameters:
        input (pathlib.Path): Filepath to input csv.
        timeframe (str): Timeframe for OHLC data (e.g., '1Min', '1H', '1D').
        start (Optional[str]): Starting UTC date before which to filter out data.
        end (Optional[str]): Ending UTC date after which to filter out data.

    Returns:
        pandas DataFrame: DataFrame containing OHLC data.
    """
    # Convert tick data to DataFrame
    df = pl.read_csv(
        input, has_header=False, new_columns=["timestamp", "price", "volume"]
    )

    # Convert timestamp column to datetime
    df = df.with_columns(
        pl.col("timestamp").apply(lambda x: datetime.utcfromtimestamp(x))
    ).sort("timestamp")

    if start is not None:
        # Filter on start
        df = df.filter(pl.col("timestamp") >= pd.to_datetime(start))
    if end is not None:
        # Filter on start
        df = df.filter(pl.col("timestamp") <= pd.to_datetime(end))

    print("Input dataframe:")
    print(df)

    # Group by the resampled timeframe and calculate OHLC
    ohlc_data = df.group_by_dynamic("timestamp", every="1h").agg(
        [
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ]
    )

    return ohlc_data


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise OSError("Input path not found")

    ohlc = tick_to_ohlc(
        input_path, timeframe=args.timeframe, start=args.start, end=args.end
    )
    ticker = input_path.stem
    # Hack to fix ticker names
    if ticker == "XBTUSD":
        ticker = "BTC/USD"
    elif ticker == "ETHUSD":
        ticker = "ETH/USD"
    elif ticker == "MATICUSD":
        ticker = "MATIC/USD"
    ohlc = ohlc.with_columns(pl.lit(ticker).alias("ticker"))

    if args.output is None:
        # Print data
        print(ohlc)
    else:
        # Output file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ohlc.write_csv(output_path)
        print(
            f"Converted {input_path} into {args.timeframe} OHLC data at {output_path}"
        )
