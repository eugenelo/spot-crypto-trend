TIMESTAMP_COL = "timestamp"
TICKER_COL = "ticker"
VOLUME_COL = "volume"
CLOSE_COL = "close"
VWAP_COL = "vwap"
PRICE_COL_SIGNAL_GEN = VWAP_COL
PRICE_COL_BACKTEST = VWAP_COL

POSITION_COL = "scaled_position"


def in_universe_excl_stablecoins(pair: str) -> bool:
    # Exclude stable coins & other untradable coins
    BLACKLISTED_TICKERS = ["PEPE"]  # Can't trade on Kraken pro for some reason
    STABLECOIN_TICKERS = ["DAI", "USDT", "EURT", "TUSD", "USDC"]
    if any([pair.startswith(ticker) for ticker in BLACKLISTED_TICKERS]):
        return False
    elif any([pair.startswith(ticker) for ticker in STABLECOIN_TICKERS]):
        return False
    return True
