from data.constants import VWAP_COL

PRICE_COL_SIGNAL_GEN = VWAP_COL
PRICE_COL_BACKTEST = VWAP_COL
RETURNS_COL = "returns"
PAST_7D_RETURNS_COL = "7d_returns"

VOLUME_ABOVE_MIN_COL = "volume_above_min"
VOLUME_BELOW_MAX_COL = "volume_below_max"
AVG_DOLLAR_VOLUME_COL = "avg_1d_dollar_volume_over_30d"
VOLUME_FILTER_COL = "filter_volume"

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
