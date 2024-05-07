import argparse
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytz
from tqdm.auto import tqdm

from data.constants import DATETIME_COL, OHLC_COLUMNS, TICKER_COL
from data.utils import fill_missing_ohlc, valid_ohlc_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Combine OHLC data into a single file")
    parser.add_argument(
        "--input_dir", "-i", required=True, type=str, help="Input directory"
    )
    parser.add_argument(
        "--data_frequency",
        "-f",
        type=str,
        required=True,
        help="Frequency of OHLC data",
    )
    parser.add_argument("--output_path", "-o", type=str, help="Output file path")
    parser.add_argument("--output_dir", "-do", type=str, help="Output directory")
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recurse through subdirectories from input directory",
    )
    parser.add_argument(
        "--drop_incomplete_rows",
        "-d",
        action="store_true",
        help="Drop incomplete OHLC rows",
    )
    parser.add_argument(
        "--auto",
        "-a",
        action="store_true",
        help=(
            "Automatic update mode - combine only the latest monthly files based on"
            " the current date"
        ),
    )
    return parser.parse_args()


def process_fileset(
    input_paths: List[Path],
    data_frequency: str,
    output_path: Optional[Path],
    utc_now_for_drop: Optional[pd.Timestamp],
):
    df_ohlc = pd.DataFrame(columns=OHLC_COLUMNS)

    for input_path in tqdm(input_paths):
        print(f"Processing {input_path}")
        with input_path.open() as fin:
            df_single = pd.read_csv(fin)
            df_ohlc = pd.concat([df_ohlc, df_single], ignore_index=True)
    df_ohlc[DATETIME_COL] = pd.to_datetime(df_ohlc[DATETIME_COL], utc=True)
    df_ohlc = df_ohlc.sort_values(by=[TICKER_COL, DATETIME_COL], ascending=True)

    # Fill in missing dates
    min_dates = (
        df_ohlc.groupby(TICKER_COL)[DATETIME_COL].min().to_frame(name="min_timestamp")
    )
    max_dates = (
        df_ohlc.groupby(TICKER_COL)[DATETIME_COL].max().to_frame(name="max_timestamp")
    )
    dates = min_dates.merge(max_dates, how="left", on=TICKER_COL)
    mIdx = pd.MultiIndex.from_frame(
        dates.apply(
            lambda x: pd.date_range(
                x["min_timestamp"], x["max_timestamp"], freq=data_frequency
            ),
            axis=1,
        )
        .explode()
        .reset_index(name=DATETIME_COL)[[TICKER_COL, DATETIME_COL]]
    )
    df_ohlc = df_ohlc.set_index([TICKER_COL, DATETIME_COL]).reindex(mIdx).reset_index()
    df_ohlc = fill_missing_ohlc(df_ohlc)

    # Check for dataframe validity (no duplicates, no missing data)
    assert valid_ohlc_dataframe(df_ohlc, freq=data_frequency), "Combined df is invalid!"

    if utc_now_for_drop is not None:
        # Drop incomplete rows
        last_complete_timestamp = (
            utc_now_for_drop - pd.to_timedelta(data_frequency)
        ).floor(freq=data_frequency)
        df_ohlc = df_ohlc.loc[df_ohlc[DATETIME_COL] <= last_complete_timestamp]

    # Reset index and reorder columns
    df_ohlc.sort_values(by=[DATETIME_COL, TICKER_COL], ascending=True, inplace=True)
    df_ohlc.index = pd.RangeIndex(len(df_ohlc.index))
    df_ohlc = df_ohlc[OHLC_COLUMNS]

    if output_path is None:
        # Print data
        print(df_ohlc)
    else:
        # Output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_ohlc.to_csv(str(output_path), index=False)
        print(f"Wrote {df_ohlc.shape} dataframe to '{output_path}'")


def get_output_path(output_dir: Path, path_start: datetime, path_end: datetime) -> Path:
    path_start_date_str = path_start.strftime("%Y-%m-%d")
    path_end_date_str = path_end.strftime("%Y-%m-%d")
    return output_dir / f"combined_{path_start_date_str}_{path_end_date_str}_OHLC.csv"


def main(args):
    assert not (
        args.auto and (args.recursive or args.drop_incomplete_rows or args.output_path)
    ), "'--auto' can only be used with '--input_dir' and '--output_dir'"

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise OSError("Input dir path not found")

    if args.auto:
        # Split filesets into common time intervals
        input_paths = sorted(list(input_dir.rglob("*.csv")))
        time_interval_to_paths = OrderedDict()
        for input_path in input_paths:
            ticker, path_start, path_end, _ = input_path.stem.split("_")
            path_start = pytz.utc.localize(datetime.strptime(path_start, "%Y-%m-%d"))
            path_end = pytz.utc.localize(datetime.strptime(path_end, "%Y-%m-%d"))
            common_paths_list = time_interval_to_paths.setdefault(
                (path_start, path_end), []
            )
            common_paths_list.append(input_path)
        time_interval_to_paths = OrderedDict(
            sorted(time_interval_to_paths.items(), key=lambda x: x[0])
        )
        # Use common drop timestamp for all output paths
        output_dir = Path(args.output_dir)
        utc_now_for_drop = pd.Timestamp.utcnow()
        write_start_date = utc_now_for_drop - timedelta(days=7)
        for time_interval, paths_list in tqdm(time_interval_to_paths.items()):
            path_start, path_end = time_interval
            output_path = get_output_path(
                output_dir=output_dir, path_start=path_start, path_end=path_end
            )
            # Only overwrite existing files with an end date within 7 days of today
            if output_path.exists():
                if write_start_date > path_end:
                    print(f"{output_path} already exists, skipping!")
                    continue
            process_fileset(
                input_paths=paths_list,
                data_frequency=args.data_frequency,
                output_path=output_path,
                utc_now_for_drop=utc_now_for_drop,
            )
    else:
        input_paths = sorted(
            list(
                input_dir.rglob("*.csv") if args.recursive else input_dir.glob("*.csv")
            )
        )
        output_path = Path(args.output_path) if args.output_path else None
        utc_now_for_drop = pd.Timestamp.utcnow() if args.drop_incomplete_rows else None
        process_fileset(
            input_paths=input_paths,
            data_frequency=args.data_frequency,
            output_path=output_path,
            utc_now_for_drop=utc_now_for_drop,
        )

    return 0


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    exit(main(args))
