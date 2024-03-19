import argparse
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional
import datetime
import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Convert tick data to OHLC data")
    parser.add_argument("--input_path", "-i", type=str, help="Input file path")
    parser.add_argument("--output_path", "-o", type=str, help="Output file path")
    parser.add_argument("--input_dir", "-di", type=str, help="Input directory")
    parser.add_argument("--output_dir", "-do", type=str, help="Output directory")
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


def tick_to_ohlc(
    input_path: Path, timeframe: str, start: Optional[str], end: Optional[str]
):
    """
    Convert tick data to OHLC data.

    Parameters:
        input_path (pathlib.Path): Filepath to input_path csv.
        timeframe (str): Timeframe for OHLC data (e.g., '1Min', '1H', '1D').
        start (Optional[str]): Starting UTC date before which to filter out data.
        end (Optional[str]): Ending UTC date after which to filter out data.

    Returns:
        pandas DataFrame: DataFrame containing OHLC data.
    """
    # Convert tick data to DataFrame
    df = pl.read_csv(
        input_path, has_header=False, new_columns=["timestamp", "price", "volume"]
    )
    df = df.with_columns(dollar_volume=pl.col("price") * pl.col("volume"))

    # Convert timestamp column to datetime
    df = df.with_columns(
        pl.col("timestamp").map_elements(
            lambda x: datetime.datetime.utcfromtimestamp(x)
        )
    ).sort("timestamp")

    if start is not None:
        # Filter on start
        df = df.filter(pl.col("timestamp") >= pd.to_datetime(start, utc=True))
    if end is not None:
        # Filter on start
        df = df.filter(pl.col("timestamp") <= pd.to_datetime(end, utc=True))

    print(f"Processing '{input_path}', shape: {df.shape}")

    # Group by the resampled timeframe and calculate OHLC
    ohlc_data = df.group_by_dynamic("timestamp", every="1h").agg(
        [
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("dollar_volume").sum().alias("dollar_volume"),
        ]
    )
    # TODO(@eugene.lo): Add vwap

    return ohlc_data


def main(
    input_path: Path,
    timeframe: str,
    start: Optional[str],
    end: Optional[str],
    output_path: Optional[Path],
):
    """
    Main wrapper.

    Parameters:
        input_path (pathlib.Path): Filepath to input csv.
        timeframe (str): Timeframe for OHLC data (e.g., '1Min', '1H', '1D').
        start (Optional[str]): Starting UTC date before which to filter out data.
        end (Optional[str]): Ending UTC date after which to filter out data.
        output_path (Optional[pathlib.Path]): Filepath to output csv.

    Returns:
        pandas DataFrame: DataFrame containing OHLC data.
    """
    ohlc = tick_to_ohlc(
        input_path=input_path, timeframe=timeframe, start=start, end=end
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

    if output_path is None:
        # Print data
        print(ohlc)
    else:
        # Output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ohlc.write_csv(output_path)
        print(
            f"Converted {input_path} into {args.timeframe} OHLC data at {output_path}"
        )


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    assert not (
        (args.input_path is None and args.input_dir is None)
        or (args.input_path is not None and args.input_dir is not None)
    ), "Exactly one of either '--input' or '--input_dir' must be specified!"

    if args.input_path is not None:
        input_path = Path(args.input_path)
        if not input_path.exists():
            raise OSError("Input path not found")
        output_path = Path(args.output_path) if args.output_path else None
        main(
            input_path=input_path,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
            output_path=output_path,
        )
    else:
        for filename in glob.glob(args.input_dir + "/*.csv"):
            input_path = Path(filename)
            output_path = (
                (Path(args.output_dir) / f"{input_path.stem}_OHLC{input_path.suffix}")
                if args.output_dir
                else None
            )
            main(
                input_path=input_path,
                timeframe=args.timeframe,
                start=args.start,
                end=args.end,
                output_path=output_path,
            )
