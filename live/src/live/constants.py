from data.constants import (
    DATETIME_COL,
    ORDER_SIDE_COL,
    ORDER_TYPE_COL,
    PRICE_COL,
    TICKER_COL,
)

# Live Trades
CURRENT_POSITION_COL = "current_position"
POSITION_DELTA_COL = "position_delta"
TARGET_DOLLAR_POSITION_COL = "target_dollar_position"
CURRENT_DOLLAR_POSITION_COL = "current_dollar_position"
TRADE_COL = "trade"

TRADE_COLUMNS = [
    CURRENT_POSITION_COL,
    POSITION_DELTA_COL,
    TARGET_DOLLAR_POSITION_COL,
    CURRENT_DOLLAR_POSITION_COL,
    TRADE_COL,
]


# Historical Trades
ID_COL = "id"
AMOUNT_COL = "amount"
COST_COL = "cost"
FEES_COL = "fees"
HISTORICAL_TRADE_COLUMNS = [
    ID_COL,
    DATETIME_COL,
    TICKER_COL,
    ORDER_TYPE_COL,
    ORDER_SIDE_COL,
    PRICE_COL,
    AMOUNT_COL,
    COST_COL,
    FEES_COL,
]

CURRENCY_COL = "currency"
BASE_CURRENCY = "USD"

# Deposits
FEE_COL = "fee"
LEDGER_DIRECTION_COL = "direction"
LEDGER_COLUMNS = [
    ID_COL,
    DATETIME_COL,
    CURRENCY_COL,
    AMOUNT_COL,
    LEDGER_DIRECTION_COL,
    FEE_COL,
]
