import vectorbt as vbt

ENTRY_TIMESTAMP_COL = "Entry Timestamp"
EXIT_TIMESTAMP_COL = "Exit Timestamp"


def get_entry_trades(pf: vbt.Portfolio):
    return pf.entry_trades.records_readable.sort_values(by=ENTRY_TIMESTAMP_COL)


def get_exit_trades(pf: vbt.Portfolio):
    exit_trades = pf.exit_trades.records_readable.sort_values(by=EXIT_TIMESTAMP_COL)
    return exit_trades.loc[exit_trades["Status"] == "Closed"]


def get_cash(pf: vbt.Portfolio):
    return pf.cash()


def get_value(pf: vbt.Portfolio):
    return pf.value()


def get_final_value(pf: vbt.Portfolio):
    return pf.final_value()


def get_returns(pf: vbt.Portfolio):
    return pf.returns()


def get_cumulative_returns(pf: vbt.Portfolio):
    return pf.cumulative_returns()


def get_annualized_return(pf: vbt.Portfolio):
    return pf.annualized_return()


def get_annualized_volatility(pf: vbt.Portfolio):
    return pf.annualized_volatility()
