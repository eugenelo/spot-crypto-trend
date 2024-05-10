import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta
from tqdm.auto import tqdm

from data.constants import ID_COL, TICK_COLUMNS, TICKER_COL, TIMESTAMP_COL
from data.utils import valid_tick_df_pandas
from logging_custom.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix write stamp alignment for tick csv files"
    )
    parser.add_argument(
        "--input_dir", "-i", type=str, required=True, help="Input file directory"
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        help="Output file directory, leave empty to overwrite input files",
    )
    parser.add_argument(
        "--write_cadence",
        "-c",
        type=str,
        required=True,
        help=(
            "Cadence with which to split files [daily, weekly, monthly, quarterly,"
            " annually]"
        ),
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recurse through subdirectories from input directory",
    )
    return parser.parse_args()


def get_subdirectories(filepaths: List[Path]) -> List[Path]:
    subdirs = set()
    for path in filepaths:
        subdirs.add(path.parent)
    return sorted(list(subdirs))


def get_output_path(
    output_dir: Path, symbol: str, write_start: datetime, write_end: datetime
) -> Path:
    symbol_str_clean = symbol.replace("/", "")
    return output_dir / Path(
        f"{symbol_str_clean}_{write_start.strftime('%Y-%m-%d')}_{write_end.strftime('%Y-%m-%d')}.csv"
    )


def get_write_start_and_dt(
    min_date: datetime, write_cadence: str
) -> Tuple[datetime, relativedelta]:
    if write_cadence == "annually":
        write_start = datetime(year=min_date.year, month=1, day=1, tzinfo=pytz.UTC)
        dt = relativedelta(years=1)
    elif write_cadence == "quarterly":
        write_start = datetime(year=min_date.year, month=1, day=1, tzinfo=pytz.UTC)
        dt = relativedelta(months=3)
    elif write_cadence == "monthly":
        write_start = datetime(
            year=min_date.year, month=min_date.month, day=1, tzinfo=pytz.UTC
        )
        dt = relativedelta(months=1)
    elif write_cadence == "weekly":
        day_offset = min_date.isoweekday() % 7
        write_start = datetime(
            year=min_date.year,
            month=min_date.month,
            day=min_date.day - day_offset,
            tzinfo=pytz.UTC,
        )
        dt = relativedelta(weeks=1)
    elif write_cadence == "daily":
        write_start = datetime(
            year=min_date.year, month=min_date.month, day=min_date.day, tzinfo=pytz.UTC
        )
        dt = relativedelta(days=1)
    else:
        raise ValueError(f"Invalid write_cadence '{write_cadence}'!")
    return write_start, dt


def process_fileset(input_paths: List[Path], output_dir: Path, write_cadence: str):
    # Read csvs to combined dataframe
    df_tick = pd.DataFrame(columns=TICK_COLUMNS)
    for input_path in input_paths:
        with input_path.open() as fin:
            df_single = pd.read_csv(fin)
            df_tick = pd.concat([df_tick, df_single], ignore_index=True)
    df_tick.sort_values(by=[ID_COL], ascending=True, inplace=True)
    # Drop duplicate transactions
    df_tick.drop_duplicates(inplace=True)
    assert valid_tick_df_pandas(
        df_tick, combined=True
    ), f"Combined tick data for {input_paths[0].parent} invalid!"

    # Rewrite dataframe to UTC monthly cadence
    symbol = df_tick[TICKER_COL].min()
    min_date = datetime.fromtimestamp(df_tick[TIMESTAMP_COL].min(), tz=pytz.UTC)
    write_start, dt = get_write_start_and_dt(
        min_date=min_date, write_cadence=write_cadence
    )
    write_end = write_start + dt
    while True:
        start_mask = df_tick[TIMESTAMP_COL] >= write_start.timestamp()
        end_mask = df_tick[TIMESTAMP_COL] < write_end.timestamp()
        df_write = df_tick.loc[(start_mask) & (end_mask)]
        if df_write.empty:
            break
        df_write = df_write.sort_values(by=[ID_COL], ascending=True)

        output_path = get_output_path(
            output_dir=output_dir,
            symbol=symbol,
            write_start=write_start,
            write_end=write_end,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_write.to_csv(output_path, index=False)
        logger.info(f"Wrote {df_write.shape} dataframe to '{output_path}'")

        write_start += dt
        write_end += dt


def main(args):
    input_dir_path = Path(args.input_dir)
    if not input_dir_path.exists():
        raise OSError("Input dir path not found")

    if not args.recursive:
        input_paths = sorted(list(input_dir_path.glob("*.csv")))
        output_dir = Path(args.output_dir) if args.output_dir else input_dir_path
        process_fileset(
            input_paths=input_paths,
            output_dir=output_dir,
            write_cadence=args.write_cadence,
        )
    else:
        input_paths = sorted(list(input_dir_path.rglob("*.csv")))
        subdirs = get_subdirectories(input_paths)
        for subdir in tqdm(subdirs):
            input_paths = sorted(list(subdir.glob("*.csv")))
            output_dir = (
                Path(args.output_dir) / subdir.relative_to(input_dir_path)
                if args.output_dir
                else subdir
            )
            process_fileset(
                input_paths=input_paths,
                output_dir=output_dir,
                write_cadence=args.write_cadence,
            )

    return 0


if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    pd.set_option("display.width", 2000)
    pd.set_option("display.precision", 3)
    pd.set_option("display.float_format", "{:.3f}".format)

    # Configure logging
    log_config_path = Path(__file__).parent / Path(
        "../../../logging_custom/logging_config/data_config.yaml"
    )
    setup_logging(config_path=log_config_path)
    logger = logging.getLogger(__name__)

    args = parse_args()
    exit(main(args))
