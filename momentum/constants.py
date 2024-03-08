# Exclude stable coins & Luna/Terra
def blacklisted(pair: str) -> bool:
    BLACKLISTED_TICKERS = ["UST", "LUNA"]
    STABLECOIN_TICKERS = ["DAI", "USDT", "EURT", "TUSD", "USDC"]

    if any([pair.startswith(ticker) for ticker in BLACKLISTED_TICKERS]):
        return True
    elif any([pair.startswith(ticker) for ticker in STABLECOIN_TICKERS]):
        return True
    return False
