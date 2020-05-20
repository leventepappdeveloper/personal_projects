"""
Author: Levente Papp
Date: 5/20/2020
"""
import MyDataReader as dr
import Helpers as hp

class MAIN_Tester():
    """
    Wrapper Class for the other individual testing classes below.
    """
    def __init__(self, ticker, startDate, endDate, interval, ma_period):
        self.ticker = ticker
        self.startDate = startDate
        self.endDate = endDate
        self.interval = interval
        self.ma_period = ma_period

        # This code is here to only have to read in the data once instead of 9 times
        self.data_reader = dr.DataReader(self.ticker, self.startDate, self.endDate, self.interval)
        self.data = self.data_reader.get_data()

    def run(self):
        df1 = simple_MA_Long_Strategy_BackTester(self.data, self.ticker, self.startDate, self.endDate,
                                                 self.interval, self.ma_period).run()
        df2 = simple_MA_Short_Strategy_BackTester(self.data, self.ticker, self.startDate, self.endDate,
                                                 self.interval, self.ma_period).run()
        df3 = simple_MA_Combined_Strategy_BackTester(self.data, self.ticker, self.startDate, self.endDate,
                                                 self.interval, self.ma_period).run()
        df4 = smoothed_MA_Long_Strategy_BackTester(self.data, self.ticker, self.startDate, self.endDate,
                                                 self.interval, self.ma_period).run()
        df5 = smoothed_MA_Short_Strategy_BackTester(self.data, self.ticker, self.startDate, self.endDate,
                                                 self.interval, self.ma_period).run()
        df6 = smoothed_MA_Combined_Strategy_BackTester(self.data, self.ticker, self.startDate, self.endDate,
                                                 self.interval, self.ma_period).run()
        df7 = exponential_MA_Long_Strategy_BackTester(self.data, self.ticker, self.startDate, self.endDate,
                                                 self.interval, self.ma_period).run()
        df8 = exponential_MA_Short_Strategy_BackTester(self.data, self.ticker, self.startDate, self.endDate,
                                                 self.interval, self.ma_period).run()
        df9 = exponential_MA_Combined_Strategy_BackTester(self.data, self.ticker, self.startDate, self.endDate,
                                                 self.interval, self.ma_period).run()

        df = df1.append(df2, ignore_index=True)
        df = df.append(df3, ignore_index=True)
        df = df.append(df4, ignore_index=True)
        df = df.append(df5, ignore_index=True)
        df = df.append(df6, ignore_index=True)
        df = df.append(df7, ignore_index=True)
        df = df.append(df8, ignore_index=True)
        df = df.append(df9, ignore_index=True)

        return df
#######################################################################################################
class smoothed_MA_Long_Strategy_BackTester():
    def __init__(self, data, ticker, startDate, endDate, interval, ma_period):
        self.ticker = ticker
        self.startDate = startDate
        self.endDate = endDate
        self.interval = interval
        self.ma_period = ma_period
        self.data = data

    def run(self):
        sma = hp.compute_smoothed_MA(self.data["Close"], self.ma_period)
        data = self.data.iloc[self.ma_period:]
        data["MA"] = sma[self.ma_period:]
        long_trades = hp.get_moving_average_trades_long_only(data)
        long_returns = hp.compute_returns(long_trades)
        strategy_name = "Long Only Smoothed MA"
        return hp.analyze_returns(strategy_name, data, long_returns, long_trades)

class smoothed_MA_Short_Strategy_BackTester():
    def __init__(self, data, ticker, startDate, endDate, interval, ma_period):
        self.ticker = ticker
        self.startDate = startDate
        self.endDate = endDate
        self.interval = interval
        self.ma_period = ma_period
        self.data = data

    def run(self):
        sma = hp.compute_smoothed_MA(self.data["Close"], self.ma_period)
        data = self.data.iloc[self.ma_period:]
        data["MA"] = sma[self.ma_period:]
        short_trades = hp.get_moving_average_trades_short_only(data)
        short_returns = hp.compute_returns(short_trades)
        strategy_name = "Short Only Smoothed MA"
        return hp.analyze_returns(strategy_name, data, short_returns, short_trades)

class smoothed_MA_Combined_Strategy_BackTester():
    def __init__(self, data, ticker, startDate, endDate, interval, ma_period):
        self.ticker = ticker
        self.startDate = startDate
        self.endDate = endDate
        self.interval = interval
        self.ma_period = ma_period
        self.data = data

    def run(self):
        sma = hp.compute_smoothed_MA(self.data["Close"], self.ma_period)
        data = self.data.iloc[self.ma_period:]
        data["MA"] = sma[self.ma_period:]
        trades = hp.get_moving_average_trades_combined(data)
        returns = hp.compute_returns(trades)
        strategy_name = "Long/Short Smoothed MA"
        return hp.analyze_returns(strategy_name, data, returns, trades)

class simple_MA_Long_Strategy_BackTester():
    def __init__(self, data, ticker, startDate, endDate, interval, ma_period):
        self.ticker = ticker
        self.startDate = startDate
        self.endDate = endDate
        self.interval = interval
        self.ma_period = ma_period
        self.data = data

    def run(self):
        sma = hp.compute_simple_MA(self.data["Close"], self.ma_period)
        data = self.data.iloc[self.ma_period:]
        data["MA"] = sma[self.ma_period:]
        long_trades = hp.get_moving_average_trades_long_only(data)
        long_returns = hp.compute_returns(long_trades)
        strategy_name = "Long Only Simple MA"
        return hp.analyze_returns(strategy_name, data, long_returns, long_trades)

class simple_MA_Short_Strategy_BackTester():
    def __init__(self, data, ticker, startDate, endDate, interval, ma_period):
        self.ticker = ticker
        self.startDate = startDate
        self.endDate = endDate
        self.interval = interval
        self.ma_period = ma_period
        self.data = data

    def run(self):
        sma = hp.compute_simple_MA(self.data["Close"], self.ma_period)
        data = self.data.iloc[self.ma_period:]
        data["MA"] = sma[self.ma_period:]
        short_trades = hp.get_moving_average_trades_short_only(data)
        short_returns = hp.compute_returns(short_trades)
        strategy_name = "Short Only Simple MA"
        return hp.analyze_returns(strategy_name, data, short_returns, short_trades)

class simple_MA_Combined_Strategy_BackTester():
    def __init__(self, data, ticker, startDate, endDate, interval, ma_period):
        self.ticker = ticker
        self.startDate = startDate
        self.endDate = endDate
        self.interval = interval
        self.ma_period = ma_period
        self.data = data

    def run(self):
        sma = hp.compute_simple_MA(self.data["Close"], self.ma_period)
        data = self.data.iloc[self.ma_period:]
        data["MA"] = sma[self.ma_period:]
        trades = hp.get_moving_average_trades_combined(data)
        returns = hp.compute_returns(trades)
        strategy_name = "Long/Short Simple MA"
        return hp.analyze_returns(strategy_name, data, returns, trades)

class exponential_MA_Long_Strategy_BackTester():
    def __init__(self, data, ticker, startDate, endDate, interval, ma_period):
        self.ticker = ticker
        self.startDate = startDate
        self.endDate = endDate
        self.interval = interval
        self.ma_period = ma_period
        self.data = data

    def run(self):
        ema = hp.compute_exponential_MA(self.data["Close"], self.ma_period)
        data = self.data.iloc[self.ma_period:]
        data["MA"] = ema[self.ma_period:]
        long_trades = hp.get_moving_average_trades_long_only(data)
        long_returns = hp.compute_returns(long_trades)
        strategy_name = "Long Only Exponential MA"
        return hp.analyze_returns(strategy_name, data, long_returns, long_trades)

class exponential_MA_Short_Strategy_BackTester():
    def __init__(self, data, ticker, startDate, endDate, interval, ma_period):
        self.ticker = ticker
        self.startDate = startDate
        self.endDate = endDate
        self.interval = interval
        self.ma_period = ma_period
        self.data = data

    def run(self):
        ema = hp.compute_exponential_MA(self.data["Close"], self.ma_period)
        data = self.data.iloc[self.ma_period:]
        data["MA"] = ema[self.ma_period:]
        short_trades = hp.get_moving_average_trades_short_only(data)
        short_returns = hp.compute_returns(short_trades)
        strategy_name = "Short Only Exponential MA"
        return hp.analyze_returns(strategy_name, data, short_returns, short_trades)

class exponential_MA_Combined_Strategy_BackTester():
    def __init__(self, data, ticker, startDate, endDate, interval, ma_period):
        self.ticker = ticker
        self.startDate = startDate
        self.endDate = endDate
        self.interval = interval
        self.ma_period = ma_period
        self.data = data

    def run(self):
        ema = hp.compute_exponential_MA(self.data["Close"], self.ma_period)
        data = self.data.iloc[self.ma_period:]
        data["MA"] = ema[self.ma_period:]
        trades = hp.get_moving_average_trades_combined(data)
        returns = hp.compute_returns(trades)
        strategy_name = "Long/Short Exponential MA"
        return hp.analyze_returns(strategy_name, data, returns, trades)