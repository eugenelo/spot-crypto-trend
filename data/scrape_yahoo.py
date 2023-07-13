import re
from io import StringIO
from datetime import datetime, timedelta

from pathlib import Path
import requests
import pandas as pd
from typing import List


class YahooFinanceHistory:
    timeout = 2
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36"
    }
    crumb_link = "https://finance.yahoo.com/quote/{0}/history?p={0}"
    crumble_regex = r'"crumb":"(.*?)"'
    quote_link = "https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={dto}&interval=1d&events=history&crumb={crumb}"

    def __init__(self, symbol, days_back=7):
        self.symbol = symbol
        self.session = requests.Session()
        self.dt = timedelta(days=days_back)

    def get_crumb(self):
        response = self.session.get(
            self.crumb_link.format(self.symbol),
            timeout=self.timeout,
            headers=self.headers,
        )
        response.raise_for_status()
        match = re.search(self.crumble_regex, response.text)
        if not match:
            raise ValueError("Could not get crumb from Yahoo Finance")
        else:
            self.crumb = match.group(1)

    def get_quote(self):
        if not hasattr(self, "crumb") or len(self.session.cookies) == 0:
            self.get_crumb()
        now = datetime.utcnow()
        dateto = int(now.timestamp())
        datefrom = int((now - self.dt).timestamp())
        url = self.quote_link.format(
            quote=self.symbol, dfrom=datefrom, dto=dateto, crumb=self.crumb
        )
        response = self.session.get(url, headers=self.headers)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), parse_dates=["Date"])


if __name__ == "__main__":
    sector = "consumer_defensive"
    sector_dir: Path = Path(f"data/{sector}")
    symbols_path: Path = Path(sector_dir, f"{sector}_symbols.txt")

    # Get symbols for sector
    symbols: List[str] = []
    with open(symbols_path, "r") as symbols_file:
        for line in symbols_file:
            # Escape comments
            if line.startswith("#"):
                continue
            symbols.append(line.strip())

    # Scrape data
    print(f"Scraping data for {len(symbols)} symbols")
    days_back = 365 * 10  # 10 years
    for symbol in symbols:
        print(f"Scraping data for {symbol}")
        df = YahooFinanceHistory(symbol, days_back=days_back).get_quote()
        out_filepath = Path(sector_dir, f"{symbol}.csv")
        out_filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_filepath, index=False)
        print(f"Saved data to {out_filepath}")
