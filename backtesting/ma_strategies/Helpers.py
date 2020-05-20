"""
Author: Levente Papp
Date: 5/20/2020
"""
import statistics
import pandas as pd

################################################################################################################
def get_moving_average_trades_long_only(data):
    """
    Find trades under long MA strategy.

    :param data: pandas dataframe
    :return: list of trades (a trade is a list itself: [entry price, exit price, "L"]
    """
    trades = []
    holding = False
    current = []
    for index, row in data[1:].iterrows():
        close = float('%.2f' % (row["Close"]))
        if not holding and row["Open"] > row["MA"] and row["Close"] > row["MA"] and row["Open"] < row["Close"]:
            holding = True
            current.append(close)

        if holding and row["Close"] < row["MA"]:
            holding = False
            current.append(close)
            current.append("L")
            trades.append(current)
            current = []
    return trades

def get_moving_average_trades_short_only(data):
    """
    Find trades under short MA strategy.

    :param data: pandas dataframe
    :return: list of trades (a trade is a list itself: [entry price, exit price, "S"]
    """
    trades = []
    holding = False
    current = []
    for index, row in data[1:].iterrows():
        close = float('%.2f' % (row["Close"]))
        if not holding and row["Open"] < row["MA"] and row["Close"] < row["MA"] and row["Open"] > row["Close"]:
            holding = True
            current.append(close)

        if holding and row["Close"] > row["MA"]:
            holding = False
            current.append(close)
            trades.append(current)
            current.append("S")
            current = []
    return trades

def get_moving_average_trades_combined(data):
    """
    Find trades under long/short MA strategy.

    :param data: pandas dataframe
    :return: list of trades (a trade is a list itself: [entry price, exit price, "L"/"S"]
    """
    trades = []
    holding_long = False
    holding_short = False
    current_long = []
    current_short = []
    for index, row in data[1:].iterrows():
        close = float('%.2f' % (row["Close"]))

        if not holding_long and row["Open"] > row["MA"] and row["Close"] > row["MA"] and row["Open"] < row["Close"]:
            holding_long = True
            current_long.append(close)

        if holding_long and row["Close"] < row["MA"]:
            holding_long = False
            current_long.append(close)
            current_long.append("L")
            trades.append(current_long)
            current_long = []

        if not holding_short and row["Open"] < row["MA"] and row["Close"] < row["MA"] and row["Open"] > row["Close"]:
            holding_short = True
            current_short.append(close)

        if holding_short and row["Close"] > row["MA"]:
            holding_short = False
            current_short.append(close)
            trades.append(current_short)
            current_short.append("S")
            current_short = []
    return trades
################################################################################################################

def compute_simple_MA(close_prices, period):
    """
    Find simple moving average of a list of close prices.

    :param close_prices: close prices column from pandas dataframe
    :param period: float; period that we are averaging over
    :return: list of floats (moving average values)
    """
    ma = list(close_prices[:period])
    for i in range(period, len(close_prices)):
        prices = close_prices[i - period:i]
        avg = float(sum(prices)) / period
        avg = float('%.2f' % (avg))
        ma.append(avg)
    return ma

def compute_smoothed_MA(close_prices, period):
    """
    Find smoothed moving average of a list of close prices.

    :param close_prices: close prices column from pandas dataframe
    :param period: float; period that we are averaging over
    :return: list of floats (moving average values)
    """
    sma = [close_prices[0]]
    for i in range(1, len(close_prices)):
        prev_sum = sma[i - 1] * period
        current_sma = float(prev_sum - sma[i - 1] + close_prices[i]) / period
        current_sma = float('%.2f' % (current_sma))
        sma.append(current_sma)
    return sma

def compute_exponential_MA(close_prices, period):
    """
    Find exponential moving average of a list of close prices.

    :param close_prices: close prices column from pandas dataframe
    :param period: float; period that we are averaging over
    :return: list of floats (moving average values)
    """
    ema = [close_prices[0]]
    multiplier = float((2 / (period + 1)))
    for i in range(1, len(close_prices)):
        current_ema = (close_prices[i] - ema[i-1]) * multiplier + ema[i-1]
        ema.append(current_ema)
    return ema

def compute_returns(trades):
    """
    Find return on each trade.

    :param trades: list of trades (a trade is a list itself: [entry price, exit price, "L"/"S"]
    :return: list of floats (returns)
    """
    returns = []
    for trade in trades:
        ret = float((trade[1] - trade[0]) / trade[0]) * 100.0
        ret = float('%.2f' % (ret))
        if trade[2] == "S":
            ret = -1*ret
        returns.append(ret)
    return returns

def average(list):
    if len(list) == 0:
        return 0
    avg = float (sum(list)/len(list))
    avg = float('%.2f' % (avg))
    return avg

def maximum(list):
    if len(list) == 0:
        return 0
    else:
        return max(list)

def minimum(list):
    if len(list) == 0:
        return 0
    else:
        return min(list)

def variance(list):
    if len(list) == 0:
        return 0
    else:
        return float('%.2f' % (statistics.variance(list)))

def profit_ratio(returns):
    if returns == []:
        return 0
    profits = 0
    for ret in returns:
        if ret > 0:
            profits += 1
    profit_ratio = float(profits) / len(returns) * 100
    profit_ratio = float('%.2f' % (profit_ratio))
    return profit_ratio

def stock_return(data):
    data_open = data["Open"][0]
    data_close = data["Close"][-1]
    stock_return = float((data_close - data_open) / data_open) * 100.0
    stock_return = float('%.2f' % (stock_return))
    return stock_return

def absolute_return(trades):
    abs_return = 0
    for trade in trades:
        ret = trade[1]-trade[0]
        if trade[2] == "S":
            ret = -1*ret
        abs_return += ret
    return float('%.2f' % (abs_return))

def analyze_returns(strategy_name, data, returns, trades):
    """
    Analyzes the performance of a given strategy.

    :param strategy_name: string (name of strategy being backtested)
    :param data: pandas dataframe
    :param returns: list of floats
    :param trades: list of trades (a trade is a list itself: [entry price, exit price, "L"/"S"]
    :return: 1-row pandas dataframe (descriptive stats of strategy)
    """
    avg_return = str(average(returns)) + "%"
    stock_ret = str(stock_return(data)) + "%"
    prof_ratio = str(profit_ratio(returns)) + "%"
    max_profit = str(maximum(returns)) + "%"
    max_loss = str(minimum(returns)) + "%"
    var = variance(returns)
    abs_return = absolute_return(trades)
    holding_return = data["Close"][-1] - data["Open"][0]
    holding_return = float('%.2f' % (holding_return))
    strat_vs_hold = float('%.2f' % (abs_return - holding_return))

    abs_return = "$"+  str(abs_return)
    holding_return = "$" + str(holding_return)
    strat_vs_hold = "$" + str(strat_vs_hold)

    row = [strategy_name, avg_return, stock_ret, prof_ratio, max_profit, max_loss, var, abs_return, holding_return,
           strat_vs_hold]
    df = pd.DataFrame([row], columns=["Strategy", "Avg_Return", "Stock_Return", "Profit_Ratio", "Max_Profit",
                                    "Max_Loss", "Variance in Returns", "Abs_Return", "Holding_Return",
                                    "Difference"])
    return df


