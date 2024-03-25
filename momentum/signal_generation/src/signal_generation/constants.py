from enum import Enum

PRICE_COLUMN = "vwap"


class SignalType(Enum):
    HistoricalReturns = "HistoricalReturns"
    Rohrbach = "Rohrbach"
