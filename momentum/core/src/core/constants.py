from data.constants import OPEN_COL, VWAP_COL

PRICE_COL_SIGNAL_GEN = VWAP_COL
PRICE_COL_BACKTEST = OPEN_COL
RETURNS_COL = "returns"
LOG_RETURNS_COL = "log_returns"
PAST_7D_RETURNS_COL = "7d_returns"

VOLUME_ABOVE_MIN_COL = "volume_above_min"
VOLUME_BELOW_MAX_COL = "volume_below_max"
AVG_DOLLAR_VOLUME_COL = "avg_1d_dollar_volume_over_30d"
VOLUME_FILTER_COL = "filter_volume"

POSITION_COL = "scaled_position"


def in_universe_excl_stablecoins(pair: str) -> bool:
    # Exclude stable coins & other untradable coins
    GEOLOCKED_TICKERS = [
        "ACA",
        "AGLD",
        "ALICE",
        "ASTR",
        "ATLAS",
        "AUDIO",
        "BONK",
        "C98",
        "CFG",
        "CSM",
        "GLMR",
        "HDX",
        "INTR",
        "JASMY",
        "KIN",
        "LMWR",
        "LUNC",
        "MC",
        "MV",
        "NMR",
        "NODL",
        "NYM",
        "ORCA",
        "OXY",
        "PARA",
        "PEPE",
        "PERP",
        "PLA",
        "POL",
        "PSTAKE",
        "PYTH",
        "RAY",
        "REQ",
        "ROOK",
        "SAMO",
        "SDN",
        "STEP",
        "SUI",
        "TEER",
        "TRIBE",
        "USTC",
        "WIF",
        "WOO",
        "XRT",
        "YGG",
    ]  # Can't trade from the US
    WRAPPED_TOKENS = ["WBTC", "WETH"]  # Duplicate with their non-wrapped counterparts
    STABLECOIN_TICKERS = [
        "DAI",
        "USDT",
        "EURT",
        "TUSD",
        "USDC",
        "PYUSD",
        "EUR",
        "GBP",
        "AUD",
    ]
    TEMPORARY_TICKERS = ["LSK"]  # Temporarily delisted from Kraken

    tickers_outside_universe = (
        GEOLOCKED_TICKERS + WRAPPED_TOKENS + STABLECOIN_TICKERS + TEMPORARY_TICKERS
    )
    if any([pair.startswith(ticker) for ticker in tickers_outside_universe]):
        return False
    return True
