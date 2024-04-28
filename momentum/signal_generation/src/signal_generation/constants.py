from enum import Enum


class SignalType(Enum):
    HistoricalReturns = "HistoricalReturns"
    Rohrbach = "Rohrbach"


def get_signal_type(params: dict) -> SignalType:
    if params["signal"].startswith("rohrbach"):
        return SignalType.Rohrbach
    raise ValueError(
        f"Unsupported 'generate_positions' argument: {params['generate_positions']}"
    )
