import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import polars as pl
import pytz
from tqdm.auto import tqdm

from data.constants import (
    CLOSE_COL,
    DOLLAR_VOLUME_COL,
    HIGH_COL,
    ID_COL,
    LOW_COL,
    OPEN_COL,
    ORDER_SIDE_COL,
    ORDER_TYPE_COL,
    PRICE_COL,
    TICK_COLUMNS,
    TICK_COLUMNS_LEGACY,
    TICK_SCHEMA_LEGACY_POLARS,
    TICK_SCHEMA_POLARS,
    TICKER_COL,
    TIMESTAMP_COL,
    VOLUME_COL,
    VWAP_COL,
)
from data.utils import valid_tick_df_polars


def parse_args():
    parser = argparse.ArgumentParser(description="Convert tick data to OHLC data")
    parser.add_argument("--input_path", "-i", type=str, help="Input file path")
    parser.add_argument("--input_dir", "-di", type=str, help="Input directory")
    parser.add_argument("--output_path", "-o", type=str, help="Output file path")
    parser.add_argument("--output_dir", "-do", type=str, help="Output directory")
    parser.add_argument(
        "--timeframe", "-t", type=str, default="1h", help="1m / 1h / 1d / etc."
    )
    parser.add_argument(
        "--auto",
        "-a",
        action="store_true",
        help=(
            "Automatic update mode - overwrite only the latest monthly file based on"
            " the current date"
        ),
    )
    parser.add_argument(
        "--combine",
        "-c",
        action="store_true",
        help="Combine input data from the same (sub)directory into single output file",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recurse through subdirectories from input directory",
    )
    parser.add_argument(
        "--overwrite",
        "-ow",
        action="store_true",
        help="Overwrite existing files (if any), otherwise will skip processing",
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


def read_tick_csv(input_path: Path) -> pl.DataFrame:
    """
    Read tick data csv into DataFrame

    Args:
        input_path (Path): Path to tick csv

    Returns:
        pl.DataFrame: DataFrame containing tick data
    """
    try:
        # Read new format (with header and id column)
        df_tick = pl.read_csv(
            input_path,
            has_header=True,
            dtypes=TICK_SCHEMA_POLARS,
        )
        if df_tick.columns != TICK_COLUMNS:
            # This is an old format file
            raise pl.exceptions.ColumnNotFoundError()
        df_tick = df_tick.drop([ORDER_SIDE_COL, ORDER_TYPE_COL])
    except pl.exceptions.ColumnNotFoundError:
        # Read old format (no header and no id column)
        df_tick = pl.read_csv(
            input_path,
            has_header=False,
            new_columns=TICK_COLUMNS_LEGACY,
            dtypes=[pl.Float64] * len(TICK_COLUMNS_LEGACY),
        )
        # Add 1-indexed id column and ticker column
        df_tick = df_tick.with_row_index(name=ID_COL, offset=1)
        ticker = get_ticker_from_filepath(input_path)
        df_tick = df_tick.with_columns(pl.lit(ticker).alias(TICKER_COL))
    return df_tick


def get_ticker_from_filepath(input_path: Path) -> str:
    """
    Get ticker name from input path (assume filename is formatted
    '{ticker}_{suffix}.csv')

    Args:
        input_path (Path): Input path

    Returns:
        str: Ticker name for path
    """
    ticker = input_path.stem.split("_")[0]
    # Should be a USD ticker
    usd_offset = -5 if ticker.endswith("PYUSD") else -3
    return f"{ticker[:usd_offset]}/{ticker[usd_offset:]}"


def get_symbol_output_dir(output_dir: Path, symbol: str) -> Path:
    symbol_str_clean = symbol.replace("/", "")
    return Path(output_dir) / Path(symbol_str_clean)


def get_output_path(input_path: Path, output_dir: Path) -> Path:
    symbol = get_ticker_from_filepath(input_path=input_path)
    return (
        get_symbol_output_dir(output_dir=output_dir, symbol=symbol)
        / f"{input_path.stem}_OHLC{input_path.suffix}"
    )


def get_combined_output_path(input_paths: List[Path], output_dir: Path) -> Path:
    input_path = input_paths[0]
    symbol = get_ticker_from_filepath(input_path=input_path)
    symbol_str_clean = symbol.replace("/", "")
    return Path(output_dir) / f"{symbol_str_clean}_OHLC{input_path.suffix}"


def get_subdirectories(filepaths: List[Path]) -> List[Path]:
    subdirs = set()
    for path in filepaths:
        subdirs.add(path.parent)
    return sorted(list(subdirs))


def tick_to_ohlc(
    df_tick: pl.DataFrame,
    timeframe: str,
    start: Optional[str],
    end: Optional[str],
):
    """
    Convert tick data to OHLC data.

    Parameters:
        df_tick (pl.DataFrame): DataFrame containing tick data
        timeframe (str): Timeframe for OHLC data (e.g., '1m', '1h', '1d').
        start (Optional[str]): Starting UTC date before which to filter out data.
        end (Optional[str]): Ending UTC date after which to filter out data.

    Returns:
        pandas DataFrame: DataFrame containing OHLC data.
    """
    # Add dollar volume column
    df_tick = df_tick.with_columns(dollar_volume=pl.col(PRICE_COL) * pl.col(VOLUME_COL))

    # Convert timestamp column to datetime
    df_tick = df_tick.with_columns(
        pl.col(TIMESTAMP_COL).map_elements(lambda x: datetime.utcfromtimestamp(x))
    ).sort(TIMESTAMP_COL)

    if start is not None:
        # Filter on start
        df_tick = df_tick.filter(
            pl.col(TIMESTAMP_COL) >= pd.to_datetime(start, utc=True)
        )
    if end is not None:
        # Filter on start
        df_tick = df_tick.filter(pl.col(TIMESTAMP_COL) <= pd.to_datetime(end, utc=True))

    # Define a custom aggregation function to compute VWAP
    def compute_vwap(args: List[pl.Series]) -> pl.Series:
        price = args[0]
        volume = args[1]
        return (price * volume).sum() / volume.sum()

    # Resample and compute OHLCV + VWAP
    df_ohlcv = df_tick.group_by_dynamic(TIMESTAMP_COL, every=timeframe).agg(
        [
            pl.col(PRICE_COL).first().alias(OPEN_COL),
            pl.col(PRICE_COL).max().alias(HIGH_COL),
            pl.col(PRICE_COL).min().alias(LOW_COL),
            pl.col(PRICE_COL).last().alias(CLOSE_COL),
            pl.col(VOLUME_COL).sum().alias(VOLUME_COL),
            pl.col(DOLLAR_VOLUME_COL).sum().alias(DOLLAR_VOLUME_COL),
            pl.map_groups(exprs=[PRICE_COL, VOLUME_COL], function=compute_vwap).alias(
                VWAP_COL
            ),
            pl.col(TICKER_COL).first().alias(TICKER_COL),
        ]
    )
    # Fill in missing dates
    df_ohlcv = df_ohlcv.upsample(time_column=TIMESTAMP_COL, every=timeframe)
    # Fill in close first to avoid having to shift
    df_ohlcv = df_ohlcv.with_columns(pl.col(CLOSE_COL).fill_null(strategy="forward"))
    df_ohlcv = df_ohlcv.with_columns(
        [
            pl.col(OPEN_COL).fill_null(pl.col(CLOSE_COL)),
            pl.col(HIGH_COL).fill_null(pl.col(CLOSE_COL)),
            pl.col(LOW_COL).fill_null(pl.col(CLOSE_COL)),
            pl.col(VOLUME_COL).fill_null(pl.lit(0)),
            pl.col(DOLLAR_VOLUME_COL).fill_null(pl.lit(0)),
            pl.col(VWAP_COL).fill_null(pl.col(CLOSE_COL)),
            pl.col(TICKER_COL).fill_null(pl.col(TICKER_COL).first()),
        ]
    )

    return df_ohlcv


def process_tick_df(
    df_tick: pl.DataFrame,
    timeframe: str,
    start: Optional[str],
    end: Optional[str],
    output_path: Optional[Path],
):
    ohlc = tick_to_ohlc(
        df_tick=df_tick,
        timeframe=timeframe,
        start=start,
        end=end,
    )

    if output_path is None:
        # Print data
        print(ohlc)
    else:
        # Output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ohlc.write_csv(output_path)
        print(
            f"Converted {df_tick.shape} dataframe into {args.timeframe} OHLC data at"
            f" {output_path}"
        )


def process_single_path(
    input_path: Path,
    timeframe: str,
    start: Optional[str],
    end: Optional[str],
    output_path: Optional[Path],
    overwrite: bool = True,
):
    """
    Process a single path.

    Parameters:
        input_path (Path): Filepath to input csv.
        timeframe (str): Timeframe for OHLC data (e.g., '1Min', '1H', '1D').
        start (Optional[str]): Starting UTC date before which to filter out data.
        end (Optional[str]): Ending UTC date after which to filter out data.
        output_path (Optional[Path]): Filepath to output csv.

    Returns:
        pandas DataFrame: DataFrame containing OHLC data.
    """
    # Check if output path already exists and overwrite is False
    if output_path is not None and output_path.exists() and not overwrite:
        print(f"Output '{output_path}' already exists, skipping '{input_path}'")
        return

    # Convert tick data to DataFrame
    df_tick = read_tick_csv(input_path)
    assert valid_tick_df_polars(df_tick, combined=False), f"{input_path} invalid!"
    process_tick_df(
        df_tick=df_tick,
        timeframe=timeframe,
        start=start,
        end=end,
        output_path=output_path,
    )


def process_multiple_paths(
    input_paths: List[Path],
    timeframe: str,
    start: Optional[str],
    end: Optional[str],
    output_path: Optional[Path],
    overwrite: bool = True,
):
    """
    Process multiple paths as a single path.

    Parameters:
        input_paths (List[Path]): List of paths to input csvs.
        timeframe (str): Timeframe for OHLC data (e.g., '1Min', '1H', '1D').
        start (Optional[str]): Starting UTC date before which to filter out data.
        end (Optional[str]): Ending UTC date after which to filter out data.
        output_path (Optional[Path]): Filepath to output csv.

    Returns:
        pandas DataFrame: DataFrame containing OHLC data.
    """
    # Check if output path already exists and overwrite is False
    if output_path is not None and output_path.exists() and not overwrite:
        print(f"Output '{output_path}' already exists, skipping '{input_paths}'")
        return

    # Combine tick data from all paths into a single dataframe
    df_tick_combined = pl.DataFrame(schema=TICK_SCHEMA_LEGACY_POLARS)
    for input_path in input_paths:
        df_tick = read_tick_csv(input_path)
        df_tick_combined = df_tick_combined.vstack(df_tick)
    df_tick_combined.rechunk()
    df_tick_combined = df_tick_combined.sort(TIMESTAMP_COL)
    assert valid_tick_df_polars(
        df_tick_combined, combined=True
    ), f"Combined tick data for {input_paths[0].parent} invalid!"

    # Validate all paths belong to same ticker
    unique_tickers = df_tick_combined.select(pl.col(TICKER_COL).unique())
    assert (
        unique_tickers.shape[0] == 1
    ), f"Tick data with mismatching tickers being combined! tickers={unique_tickers}"

    process_tick_df(
        df_tick=df_tick_combined,
        timeframe=timeframe,
        start=start,
        end=end,
        output_path=output_path,
    )


@dataclass
class FailedJob:
    input_path: Optional[str]
    input_dir: Optional[str]
    output_path: Optional[str]
    timeframe: str
    start: Optional[str]
    end: Optional[str]
    overwrite: bool


def filter_auto_update_paths(path: Path, write_start_date: datetime) -> bool:
    """Filter function for `--auto` input paths

    Filter paths to only those with an end date within 7 days of write_start_date.

    Args:
        path (Path): _description_
        write_start_date (datetime): _description_

    Returns:
        bool: True if path should be filtered out from processing
    """
    ticker, path_start, path_end = path.stem.split("_")
    path_start = pytz.utc.localize(datetime.strptime(path_start, "%Y-%m-%d"))
    path_end = pytz.utc.localize(datetime.strptime(path_end, "%Y-%m-%d"))
    return write_start_date > path_end


def auto_update(input_dir: Path, output_dir: Path, timeframe: str) -> List[FailedJob]:
    today = pytz.utc.localize(datetime.now())
    write_start_date = today - timedelta(days=7)
    overwrite = True
    failed_jobs: List[FailedJob] = []
    input_paths = sorted(list(input_dir.rglob("*.csv")))
    for input_path in tqdm(input_paths):
        output_path = get_output_path(input_path=input_path, output_dir=output_dir)
        # Only overwrite existing files with an end date within 7 days of today
        if output_path.exists():
            if filter_auto_update_paths(
                path=input_path, write_start_date=write_start_date
            ):
                print(f"{output_path} already exists, skipping!")
                continue
        try:
            process_single_path(
                input_path=input_path,
                timeframe=timeframe,
                start=None,
                end=None,
                output_path=output_path,
                overwrite=overwrite,
            )
        except Exception as e:
            print(e)
            failed_jobs.append(
                FailedJob(
                    input_path=input_path,
                    input_dir=None,
                    output_path=output_path,
                    timeframe=args.timeframe,
                    start=None,
                    end=None,
                    overwrite=overwrite,
                )
            )
            continue
    return failed_jobs


def main(args):
    assert not (
        (args.input_path is None and args.input_dir is None)
        or (args.input_path is not None and args.input_dir is not None)
    ), "Exactly one of either '--input_path' or '--input_dir' must be specified!"
    assert not (
        args.input_path is not None and (args.combine or args.recursive)
    ), "'--combine' and '--recursive' can only be specified with '--input_dir'!"
    assert not (
        args.output_path is not None and args.output_dir is not None
    ), "At most one of either '--output_path' or '--output_dir' must be specified!"
    assert not (
        args.auto
        and (
            args.combine
            or args.recursive
            or args.overwrite
            or args.start
            or args.end
            or args.input_path
            or args.output_path
        )
    ), "'--auto' can only be used with '--input_dir' and '--output_dir' (optional)"

    if args.auto:
        # Automatic update
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise OSError("Input dir path not found")
        output_dir = Path(args.output_dir) if args.output_dir else None
        if not output_dir:
            raise ValueError("'--output_dir' not specified, required for '--auto'")
        elif not output_dir.exists():
            raise OSError("Output dir path not found")
        failed_jobs = auto_update(
            input_dir=input_dir,
            output_dir=output_dir,
            timeframe=args.timeframe,
        )
    else:
        if args.input_path is not None:
            # Process single path
            input_path = Path(args.input_path)
            if not input_path.exists():
                raise OSError("Input path not found")
            output_path = None
            if args.output_path is not None:
                output_path = Path(args.output_path)
            elif args.output_dir is not None:
                output_path = get_output_path(
                    input_path=input_path, output_dir=Path(args.output_dir)
                )
            process_single_path(
                input_path=input_path,
                timeframe=args.timeframe,
                start=args.start,
                end=args.end,
                output_path=output_path,
                overwrite=args.overwrite,
            )
        else:
            input_dir = Path(args.input_dir)
            if not input_dir.exists():
                raise OSError("Input dir path not found")

            failed_jobs: List[FailedJob] = []
            if not args.combine:
                # Process each path into a separate OHLC output
                input_paths = sorted(
                    list(
                        input_dir.rglob("*.csv")
                        if args.recursive
                        else input_dir.glob("*.csv")
                    )
                )
                for input_path in tqdm(input_paths):
                    output_path = (
                        get_output_path(
                            input_path=input_path, output_dir=Path(args.output_dir)
                        )
                        if args.output_dir
                        else None
                    )
                    try:
                        process_single_path(
                            input_path=input_path,
                            timeframe=args.timeframe,
                            start=args.start,
                            end=args.end,
                            output_path=output_path,
                            overwrite=args.overwrite,
                        )
                    except Exception as e:
                        print(e)
                        failed_jobs.append(
                            FailedJob(
                                input_path=input_path,
                                input_dir=None,
                                output_path=output_path,
                                timeframe=args.timeframe,
                                start=args.start,
                                end=args.end,
                                overwrite=args.overwrite,
                            )
                        )
                        continue
            else:
                if not args.recursive:
                    # Process all paths in the input directory into a single OHLC output
                    input_paths = sorted(list(input_dir.glob("*.csv")))
                    output_path = None
                    if args.output_path is not None:
                        output_path = Path(args.output_path)
                    elif args.output_dir is not None:
                        output_path = get_combined_output_path(
                            input_paths=input_paths, output_dir=Path(args.output_dir)
                        )
                    process_multiple_paths(
                        input_paths,
                        timeframe=args.timeframe,
                        start=args.start,
                        end=args.end,
                        output_path=output_path,
                        overwrite=args.overwrite,
                    )
                else:
                    # Recursively process paths in subdirs into separate OHLC outputs
                    input_paths = sorted(list(input_dir.rglob("*.csv")))
                    subdirs = get_subdirectories(input_paths)
                    for subdir in tqdm(subdirs):
                        input_paths = sorted(list(subdir.glob("*.csv")))
                        output_path = (
                            get_combined_output_path(
                                input_paths=input_paths,
                                output_dir=Path(args.output_dir),
                            )
                            if args.output_dir
                            else None
                        )
                        try:
                            process_multiple_paths(
                                input_paths,
                                timeframe=args.timeframe,
                                start=args.start,
                                end=args.end,
                                output_path=output_path,
                                overwrite=args.overwrite,
                            )
                        except Exception as e:
                            print(e)
                            failed_jobs.append(
                                FailedJob(
                                    input_path=None,
                                    input_dir=subdir,
                                    output_path=output_path,
                                    timeframe=args.timeframe,
                                    start=args.start,
                                    end=args.end,
                                    overwrite=args.overwrite,
                                )
                            )
                            continue

    if len(failed_jobs) > 0:
        print("Failed Jobs:")
        for failed_job in failed_jobs:
            print(f"\t- {failed_job}")
        return 1
    return 0


if __name__ == "__main__":
    args = parse_args()
    exit(main(args))
