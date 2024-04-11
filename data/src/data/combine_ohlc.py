import argparse
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
import glob
from tqdm.auto import tqdm

from data.constants import TIMESTAMP_COL, TICKER_COL, OHLC_COLUMNS


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
    df_ohlc = pd.DataFrame(columns=OHLC_COLUMNS)

    for filename in tqdm(sorted(glob.glob(args.input_dir + "/*.csv"))):
        input_path = Path(filename)
        print(f"Processing {input_path}")

        with input_path.open() as fin:
            df_single = pd.read_csv(fin)
            df_ohlc = pd.concat([df_ohlc, df_single], ignore_index=True)

    # Remove duplicates
    df_ohlc[TIMESTAMP_COL] = pd.to_datetime(df_ohlc[TIMESTAMP_COL], utc=True)
    df_ohlc.drop_duplicates(subset=[TIMESTAMP_COL, TICKER_COL], inplace=True)
    df_ohlc.sort_values(by=[TIMESTAMP_COL, TICKER_COL], ascending=True, inplace=True)
    df_ohlc.index = pd.RangeIndex(len(df_ohlc.index))

    # Output file
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_ohlc.to_csv(str(output_path), index=False)
    print(f"Converted files from '{input_dir}' into final file '{output_path}'")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    main(args.input_dir, args.output_path)
