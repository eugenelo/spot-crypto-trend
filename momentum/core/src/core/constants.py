def in_universe_excl_stablecoins(pair: str) -> bool:
    # Exclude stable coins & other untradable coins
    BLACKLISTED_TICKERS = ["PEPE"]  # Can't trade on Kraken pro for some reason
    STABLECOIN_TICKERS = ["DAI", "USDT", "EURT", "TUSD", "USDC"]
    if any([pair.startswith(ticker) for ticker in BLACKLISTED_TICKERS]):
        return False
    elif any([pair.startswith(ticker) for ticker in STABLECOIN_TICKERS]):
        return False
    return True


def in_shitcoin_trending_universe(pair: str) -> bool:
    WHITELISTED_TICKERS = [
        "ADA/USD",
        "ALCX/USD",
        "ALGO/USD",
        "ALICE/USD",
        "ANT/USD",
        "APE/USD",
        "ATLAS/USD",
        "ATOM/USD",
        "AUDIO/USD",
        "AVAX/USD",
        "BICO/USD",
        "BTC/USD",
        "DOT/USD",
        "ETH/USD",
        "FARM/USD",
        "FIDA/USD",
        "FLOW/USD",
        "FTM/USD",
        "GALA/USD",
        "GLMR/USD",
        "GNO/USD",
        "GRT/USD",
        "ICP/USD",
        "ICX/USD",
        "IMX/USD",
        "JASMY/USD",
        "KAR/USD",
        "KEEP/USD",
        "KINT/USD",
        "KP3R/USD",
        "KSM/USD",
        "LTC/USD",
        "LUNC/USD",
        "MC/USD",
        "MOVR/USD",
        "MSOL/USD",
        "MULTI/USD",
        "NANO/USD",
        "NEAR/USD",
        "OMG/USD",
        "POLIS/USD",
        "POLS/USD",
        "POWR/USD",
        "PSTAKE/USD",
        "RARE/USD",
        "RAY/USD",
        "REP/USD",
        "RLC/USD",
        "RNDR/USD",
        "SAMO/USD",
        "SBR/USD",
        "SC/USD",
        "SCRT/USD",
        "STEP/USD",
        "SUPER/USD",
        "TEER/USD",
        "TVK/USD",
        "UMA/USD",
        "WAVES/USD",
        "WOO/USD",
        "XRT/USD",
        "XTZ/USD",
    ]
    return pair in WHITELISTED_TICKERS


def in_mature_trending_universe(pair: str) -> bool:
    WHITELISTED_TICKERS = [
        "ADA/USD",
        "BCH/USD",
        "BTC/USD",
        "DOGE/USD",
        "ETH/USD",
        "SOL/USD",
        "XLM/USD",
        "XTZ/USD",
    ]
    return pair in WHITELISTED_TICKERS
