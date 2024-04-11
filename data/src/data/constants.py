import polars as pl

# OHLC data
TIMESTAMP_COL = "timestamp"
OPEN_COL = "open"
HIGH_COL = "high"
LOW_COL = "low"
CLOSE_COL = "close"
VWAP_COL = "vwap"
VOLUME_COL = "volume"
DOLLAR_VOLUME_COL = "dollar_volume"
TICKER_COL = "ticker"

OHLC_COLUMNS = [
    TIMESTAMP_COL,
    OPEN_COL,
    HIGH_COL,
    LOW_COL,
    CLOSE_COL,
    VWAP_COL,
    VOLUME_COL,
    DOLLAR_VOLUME_COL,
    TICKER_COL,
]

# Tick data
PRICE_COL = "price"
ID_COL = "id"
ORDER_SIDE_COL = "side"
ORDER_TYPE_COL = "type"
TICK_COLUMNS = [
    ID_COL,
    TIMESTAMP_COL,
    PRICE_COL,
    VOLUME_COL,
    ORDER_SIDE_COL,
    ORDER_TYPE_COL,
    TICKER_COL,
]
TICK_COLUMNS_LEGACY = [TIMESTAMP_COL, PRICE_COL, VOLUME_COL]
TICK_SCHEMA_POLARS = {
    ID_COL: pl.UInt32,
    TIMESTAMP_COL: pl.Float64,
    PRICE_COL: pl.Float64,
    VOLUME_COL: pl.Float64,
    ORDER_SIDE_COL: pl.String,
    ORDER_TYPE_COL: pl.String,
    TICKER_COL: pl.String,
}
TICK_SCHEMA_LEGACY_POLARS = {
    ID_COL: pl.UInt32,
    TIMESTAMP_COL: pl.Float64,
    PRICE_COL: pl.Float64,
    VOLUME_COL: pl.Float64,
    TICKER_COL: pl.String,
}
NUMERIC_COLUMNS = [ID_COL, TIMESTAMP_COL, PRICE_COL, VOLUME_COL]

NUM_RETRY_ATTEMPTS = 10
EPS_MS = 1  # [milliseconds]
EPS_NS = 10  # [nanoseconds]
