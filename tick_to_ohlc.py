import argparse
import pandas as pd
from pathlib import Path
from typing import Optional


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
        tick_data (list of tuples): List of tuples containing (timestamp, price, volume).
        timeframe (str): Timeframe for OHLC data (e.g., '1Min', '1H', '1D').
        start (Optional[str]): Starting UTC date before which to filter out data.
        end (Optional[str]): Ending UTC date after which to filter out data.

    Returns:
        pandas DataFrame: DataFrame containing OHLC data.
    """
    # Convert tick data to DataFrame
    with input.open() as fin:
        df = pd.read_csv(fin, header=None, names=["Timestamp", "Price", "Volume"])

    # Convert timestamp to datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")

    if start is not None:
        # Filter on start
        df = df[df["Timestamp"] >= pd.to_datetime(start)]
    if end is not None:
        # Filter on start
        df = df[df["Timestamp"] <= pd.to_datetime(end)]

    # Resample data to desired timeframe and calculate OHLC
    ohlc_data = (
        df.set_index("Timestamp")
        .resample(timeframe)
        .agg({"Price": "ohlc", "Volume": "sum"})
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
    if ticker == "XXBTZUSD":
        ticker = "BTC/USD"
    elif ticker == "XETHZUSD":
        ticker = "ETH/USD"
    elif ticker == "MATICUSD":
        ticker = "MATIC/USD"
    ohlc["ticker"] = ticker

    if args.output is None:
        # Print data
        print(ohlc)
    else:
        # Output file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ohlc.to_csv(str(output_path))
        print(
            f"Converted {input_path} into {args.timeframe} OHLC data at {output_path}"
        )
