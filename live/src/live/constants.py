from datetime import datetime

import pytz

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
TRADE_DOLLAR_COL = "trade_dollar"
TRADE_AMOUNT_COL = "trade_amount"
CURRENT_PRICE_COL = "current_price"
CURRENT_AMOUNT_COL = "current_amount"

TRADE_COLUMNS = [
    CURRENT_POSITION_COL,
    POSITION_DELTA_COL,
    TARGET_DOLLAR_POSITION_COL,
    CURRENT_DOLLAR_POSITION_COL,
    TRADE_DOLLAR_COL,
    CURRENT_PRICE_COL,
    CURRENT_AMOUNT_COL,
    TRADE_AMOUNT_COL,
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

# Trade Execution
MAX_ACCEPTABLE_SLIPPAGE = 0.005
MAX_SINGLE_TRADE_SIZE = 500  # [BASE_CURRENCY]
MARKET_ORDER_TIMEOUT_TIME = 10  # [seconds]
LIMIT_ORDER_TIMEOUT_TIME = 3600  # [seconds]
LIMIT_THEN_MARKET_SWITCH_TIME = 7200  # [seconds]
TRADE_EXECUTION_PAUSE_INTERVAL = 1  # [seconds]
UPDATE_TRADES_INTERVAL = 0  # [seconds]

# Pnl
PNL_DATA_FETCH_START_DATE = datetime(2024, 1, 1, tzinfo=pytz.UTC)
