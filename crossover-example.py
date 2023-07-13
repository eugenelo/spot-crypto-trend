from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, GOOG
from data import GM, FORD, AAPL


class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)  # 10 day moving average
        self.ma2 = self.I(SMA, price, 20)  # 20 day moving average

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


if __name__ == "__main__":
    bt = Backtest(AAPL, SmaCross, commission=0.002, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    bt.plot()
