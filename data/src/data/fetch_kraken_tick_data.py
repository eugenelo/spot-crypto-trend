import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import time as datetime_time
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta
from tqdm.auto import tqdm

from data.constants import (
    EPS_MS,
    EPS_NS,
    ID_COL,
    NUM_RETRY_ATTEMPTS,
    NUMERIC_COLUMNS,
    ORDER_SIDE_COL,
    ORDER_TYPE_COL,
    PRICE_COL,
    TICKER_COL,
    TIMESTAMP_COL,
    VOLUME_COL,
)
from data.utils import (
    get_unified_symbols,
    get_usd_symbols,
    interpolate_missing_ids,
    valid_tick_df_pandas,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch Kraken tick data")
    parser.add_argument("--output_dir", "-o", type=str, help="Output file directory")
    parser.add_argument("--ticker", "-t", type=str, help="Specific ticker")
    parser.add_argument(
        "--lookback_days", "-l", type=int, help="Lookback period in days"
    )
    parser.add_argument(
        "--since", "-s", type=float, help="Start Unix timstamp in seconds"
    )
    parser.add_argument(
        "--from_latest",
        "-f",
        action="store_true",
        help=(
            "Fetch tick data from the latest available timestamp (per ticker)."
            "Existing files should be stored under `{--output_dir}/{ticker}/`"
        ),
    )
    parser.add_argument("--end", "-e", type=float, help="End Unix timestamp in seconds")
    parser.add_argument(
        "--chunk_size",
        "-c",
        type=int,
        default=312500,
        help=(
            "Maximum number of trades to try fetching at a single time, default = 10MB"
        ),
    )
    return parser.parse_args()


class KrakenTick(ccxt.kraken):
    def fetch_trades(
        self,
        symbol: str,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[dict] = None,
    ):
        """
        get the list of most recent trades for a particular symbol
        :see: https://docs.kraken.com/rest/#tag/Spot-Market-Data/operation/getRecentTrades
        :param str symbol: unified symbol of the market to fetch trades for
        :param int [since]: timestamp in ns of the earliest trade to fetch
        :param int [limit]: the maximum amount of trades to fetch
        :param dict [params]: extra parameters specific to the exchange API endpoint
        :returns Trade[]: a list of `trade structures <https://docs.ccxt.com/#/?id=public-trades>`
        """  # noqa: B950
        self.load_markets()
        market = self.market(symbol)
        id = market["id"]
        request = {
            "pair": id,
        }
        # https://support.kraken.com/hc/en-us/articles/218198197-How-to-pull-all-trade-data-using-the-Kraken-REST-API
        # https://github.com/ccxt/ccxt/issues/5677
        if since is not None:
            # request['since'] = str(since) + "000000"
            request["since"] = str(since)  # expected to be in nanoseconds
        if limit is not None:
            request["count"] = limit

        if params is not None:
            request = self.extend(request, params)
        response = self.publicGetTrades(request)
        #
        #     {
        #         "error": [],
        #         "result": {
        #             "XETHXXBT": [
        #                 ["0.032310","4.28169434",1541390792.763,"s","l",""]
        #             ],
        #             "last": "1541439421200678657"
        #         }
        #     }
        #
        result = response["result"]
        trades = result[id]
        # trades is a sorted array: last(most recent trade) goes last
        length = len(trades)
        if length <= 0:
            return []
        lastTrade = trades[length - 1]
        lastTradeId = self.safe_string(result, "last")
        lastTrade.append(lastTradeId)
        trades[length - 1] = lastTrade
        return self.parse_trades(trades, market, since=None, limit=limit)


def get_symbol_output_dir(output_dir: Path, symbol: str) -> Path:
    symbol_str_clean = symbol.replace("/", "")
    return Path(output_dir) / Path(symbol_str_clean)


def get_output_path(
    output_dir: Path, symbol: str, write_start: datetime, write_end: datetime
) -> Path:
    symbol_str_clean = symbol.replace("/", "")
    return get_symbol_output_dir(output_dir=output_dir, symbol=symbol) / Path(
        f"{symbol_str_clean}_{write_start.strftime('%Y-%m-%d')}_{write_end.strftime('%Y-%m-%d')}.csv"
    )


def set_since_ns_from_latest(output_dir: Path, symbol: str) -> int:
    # Get start stamp for ticker from existing data
    since_ns = 0
    output_dir = get_symbol_output_dir(output_dir=output_dir, symbol=symbol)
    if output_dir.exists():
        paths_for_symbol = [path for path in output_dir.glob("*.csv")]
        if len(paths_for_symbol) > 0:
            latest_path = sorted(paths_for_symbol)[-1]
            df_latest_output = pd.read_csv(latest_path)
            if not df_latest_output.empty:
                # Use last timestamp from latest data minus a small buffer
                since = df_latest_output.sort_values(by=TIMESTAMP_COL, ascending=True)[
                    TIMESTAMP_COL
                ].tail(1).iloc[0] - (EPS_MS / 1000)
                since_ns = int(since * 1e9)
    return since_ns


def fetch_tick_data(
    kraken: ccxt.kraken,
    symbol: str,
    since_ns: int,
    end_ns: int,
    chunk_size: int,
) -> Tuple[pd.DataFrame, int]:
    """
    Fetch tick data for symbol from exchange

    Args:
        kraken (ccxt.kraken): Kraken exchange API
        symbol (str): Symbol to fetch
        since (int): Unix timestamp in nanoseconds
        end (int): Unix timestamp in nanoseconds
        chunk_size (int): Max number of rows

    Returns:
        pd.DataFrame: DataFrame containing tick data
        int: Unix stamp in nanoseconds containing the `last` parameter
             value (see https://support.kraken.com/hc/en-us/articles/218198197-How-to-retrieve-historical-time-and-sales-trading-history-using-the-REST-API-Trades-endpoint-)
    """  # noqa: B950
    # Stream tick data
    tick_data = {
        ID_COL: [],
        TIMESTAMP_COL: [],
        PRICE_COL: [],
        VOLUME_COL: [],
        ORDER_SIDE_COL: [],
        ORDER_TYPE_COL: [],
    }
    last_trade_id = None
    last_since_ns = None
    while since_ns < end_ns:
        fetched_trades: bool = False
        for _attempt in range(NUM_RETRY_ATTEMPTS):
            try:
                trades = kraken.fetch_trades(symbol, since=since_ns)
                fetched_trades = True
            except ccxt.NetworkError as e:
                print(f"{symbol} failed due to a network error: {str(e)}. Retrying!")
                continue
            except ccxt.ExchangeError as e:
                print(f"{symbol} failed due to exchange error: {str(e)}. Skipping!")
                break
            except Exception as e:
                print(f"{symbol} failed with unexpected error: {str(e)}. Skipping!")
                break
            else:
                break
        if not fetched_trades:
            break
        if len(trades) == 0:
            # No trades, done
            break
        elif (
            last_trade_id is not None
            and last_trade_id != 0
            and trades[-1][ID_COL] == last_trade_id
        ):
            # No new data, done
            break

        for trade in trades:
            tick_data[ID_COL].append(int(trade[ID_COL]))  # May be 0, which is invalid
            timestamp = float(trade["info"][2])  # [sec]
            tick_data[TIMESTAMP_COL].append(timestamp)
            tick_data[PRICE_COL].append(float(trade[PRICE_COL]))
            tick_data[VOLUME_COL].append(float(trade["amount"]))
            tick_data[ORDER_SIDE_COL].append(trade[ORDER_SIDE_COL])
            tick_data[ORDER_TYPE_COL].append(trade[ORDER_TYPE_COL])

        # Update search state
        last_trade_id = tick_data[ID_COL][-1]
        last_since_ns = since_ns
        try:
            since_ns = int(trades[-1]["info"][7])
        except IndexError:
            # Occasionally, the nanosecond field will be missing. Take from the seconds
            # field instead and decrement by eps to avoid missing trades.
            print(
                "WARNING: Couldn't get 'last'! Setting to timestamp from latest trade."
            )
            since_ns = int(tick_data[TIMESTAMP_COL][-1] * 1e9) - EPS_NS

        if last_since_ns == since_ns:
            # Repeating query stamp, done
            break
        elif len(tick_data[ID_COL]) >= chunk_size:
            # Chunk size exceeded, done
            break

    # Convert to dataframe
    df_tick = pd.DataFrame.from_dict(tick_data)
    for col in NUMERIC_COLUMNS:
        df_tick[col] = df_tick[col].map(pd.to_numeric)
    df_tick[TICKER_COL] = symbol
    return df_tick, since_ns


@dataclass
class FailedJob:
    symbol: str
    since_ns: int
    end_ns: int
    chunk_size: int


def main(args):
    # Initialize the Kraken exchange
    kraken = KrakenTick()

    # Get ccxt symbols
    if args.ticker is not None:
        symbols = get_unified_symbols(kraken, tickers=[args.ticker])
    else:
        symbols = get_usd_symbols(kraken)
    print(f"{len(symbols)} valid USD pairs")

    num_input_methods = (
        int(args.lookback_days is not None)
        + int(args.since is not None)
        + int(args.from_latest)
    )
    assert num_input_methods == 1, (
        "Must specify starting stamp via exactly one of ['--lookback_days', '--since',"
        " '--from_latest']"
    )

    # Get starting query stamp
    if args.lookback_days is not None:
        lookback = timedelta(days=args.lookback_days)
        start_date = datetime.combine(datetime.now() - lookback, datetime_time())
        since = float(kraken.parse8601(start_date.isoformat()) / 1000)
    elif args.since is not None:
        since = args.since
    elif args.from_latest:
        # Set since separately for each ticker
        since = 0
    # Get ending query stamp
    if args.end is not None:
        end = args.end
    else:
        # Don't add any buffer, otherwise symbols queried later will
        # contain more data than symbols queried earlier
        end = float(kraken.parse8601(datetime.now().isoformat()) / 1000)
    # Convert start/end timestamps to ns.
    since_ns = int(since * 1e9)
    end_ns = int(end * 1e9)

    # Track query stamp for symbol in case chunking is needed
    failed_jobs = []
    for symbol in tqdm(symbols):
        if args.from_latest:
            since_ns = set_since_ns_from_latest(
                output_dir=args.output_dir, symbol=symbol
            )
            since = float(since_ns) * 1e-9
        print(
            f"Fetching data for {symbol} from {datetime.fromtimestamp(since)} to"
            f" {datetime.fromtimestamp(end)}"
        )

        query_ns = since_ns
        last_trade_id = None
        while query_ns < end_ns:
            t0 = time.time()

            df_tick, last_ns = fetch_tick_data(
                kraken,
                symbol=symbol,
                since_ns=query_ns,
                end_ns=end_ns,
                chunk_size=args.chunk_size,
            )

            if df_tick.empty:
                # No new trades
                break
            elif last_trade_id is not None:
                if last_trade_id == df_tick[ID_COL].max():
                    # No new trades
                    break
                # Check for skipped trades
                if last_trade_id < df_tick[ID_COL].min() - 1:
                    print(
                        f"Skipped trades for {symbol}! since_ns={query_ns},"
                        f" last_trade_id={last_trade_id},"
                        f" new_min_trade_id={df_tick[ID_COL].min()}"
                    )
                    failed_jobs.append(
                        FailedJob(
                            symbol=symbol,
                            since_ns=query_ns,
                            end_ns=end_ns,
                            chunk_size=args.chunk_size,
                        )
                    )
                    break
            # Drop duplicate rows. Use full column set since id can still be 0
            df_tick.drop_duplicates(inplace=True)
            t1 = time.time()

            min_date = datetime.fromtimestamp(df_tick[TIMESTAMP_COL].min(), tz=pytz.UTC)
            max_date = datetime.fromtimestamp(df_tick[TIMESTAMP_COL].max(), tz=pytz.UTC)
            print(
                f"Fetched {len(df_tick)} trades from {min_date} to {max_date} in"
                f" {t1-t0:.2f} seconds"
            )

            # Try to correct trade_ids of value 0
            if (df_tick[ID_COL] == 0).any():
                df_tick.loc[df_tick[ID_COL] == 0, ID_COL] = np.nan

                try:
                    df_tick[ID_COL] = interpolate_missing_ids(df_tick[ID_COL])
                except RuntimeError:
                    # Interpolation failed
                    pass
                if df_tick[ID_COL].isna().any():
                    print(
                        "Failed to correct trade_ids of 0. Try fetching the data again"
                        f" with a larger '--chunk_size'\n\t- symbol={symbol},"
                        f" since_ns={query_ns}, end_ns={end_ns},"
                        f" chunk_size={args.chunk_size}"
                    )
                    failed_jobs.append(
                        FailedJob(
                            symbol=symbol,
                            since_ns=query_ns,
                            end_ns=end_ns,
                            chunk_size=args.chunk_size,
                        )
                    )
                    break

            if not valid_tick_df_pandas(df=df_tick, combined=False):
                failed_jobs.append(
                    FailedJob(
                        symbol=symbol,
                        since_ns=query_ns,
                        end_ns=end_ns,
                        chunk_size=args.chunk_size,
                    )
                )
                break

            # Separate tick data into monthly files for ease of processing
            if args.output_dir is not None:
                write_start = datetime(
                    year=min_date.year, month=min_date.month, day=1, tzinfo=pytz.UTC
                )
                dt = relativedelta(months=1)
                write_end = write_start + dt

                while write_start.timestamp() <= df_tick[TIMESTAMP_COL].max():
                    start_mask = df_tick[TIMESTAMP_COL] >= write_start.timestamp()
                    end_mask = df_tick[TIMESTAMP_COL] < write_end.timestamp()
                    df_write = df_tick.loc[(start_mask) & (end_mask)]
                    if df_write.empty:
                        write_start += dt
                        write_end += dt
                        continue

                    output_path = get_output_path(
                        output_dir=args.output_dir,
                        symbol=symbol,
                        write_start=write_start,
                        write_end=write_end,
                    )
                    if output_path.exists():
                        # Load existing output file to filter duplicates and reindex
                        df_existing_output = pd.read_csv(output_path)
                        assert df_write.columns.equals(df_existing_output.columns)
                        for col in NUMERIC_COLUMNS:
                            df_existing_output[col] = df_existing_output[col].map(
                                pd.to_numeric
                            )
                        df_write = pd.concat([df_existing_output, df_write])
                        df_write.sort_values(by=[ID_COL], ascending=True, inplace=True)
                        df_write.drop_duplicates(subset=[ID_COL], inplace=True)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    df_write.to_csv(output_path, index=False)
                    print(f"Wrote {df_write.shape} dataframe to '{output_path}'")

                    write_start += dt
                    write_end += dt
            else:
                print(df_tick)

            query_ns = last_ns
            last_trade_id = df_tick[ID_COL].max()

    if len(failed_jobs) > 0:
        print("Failed Jobs:")
        for failed_job in failed_jobs:
            print(f"\t- {failed_job}")
        return 1
    return 0


if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    pd.set_option("display.width", 2000)
    pd.set_option("display.precision", 3)
    pd.set_option("display.float_format", "{:.3f}".format)

    args = parse_args()
    exit(main(args))
