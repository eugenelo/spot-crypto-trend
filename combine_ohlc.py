import argparse
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Combine OHLC data into a single file")
    parser.add_argument(
        "--input_dir", "-i", required=True, type=str, help="Input directory"
    )
    parser.add_argument(
        "--output_path", "-o", required=True, type=str, help="Output file path"
    )
    return parser.parse_args()


def main(
    input_dir: Path,
    output_path: Path,
):
    """
    Main wrapper.

    Parameters:
        input_dir (pathlib.Path): Input file directory
        output_path (Optional[pathlib.Path]): Filepath to output csv.
    """
    ohlc_final = pd.DataFrame(
        columns=["timestamp", "open", "high", "low", "close", "volume", "ticker"]
    )

    for filename in glob.glob(args.input_dir + "/*.csv"):
        if "USD_" not in filename:
            # Only process USD pairs
            continue

        input_path = Path(filename)
        print(f"Processing {input_path}")

        with input_path.open() as fin:
            df = pd.read_csv(fin)
            ohlc_final = pd.concat([ohlc_final, df], ignore_index=True)

    # Output file
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ohlc_final.to_csv(str(output_path))
    print(f"Converted files from '{input_dir}' into final file '{output_path}'")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    main(args.input_dir, args.output_path)
