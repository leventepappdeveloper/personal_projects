"""
Author: Levente Papp
Date: 5/20/2020
"""
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

class DataReader():
    """
    Class that reads data from the Yahoo Finance API and returns a pandas dataframe.
    """
    def __init__(self, ticker, startDate, endDate, interval):
        self.ticker = ticker
        self.startDate = startDate
        self.endDate = endDate
        self.interval = interval

    def get_data(self):
        data = pdr.get_data_yahoo(self.ticker, start=self.startDate, end=self.endDate, interval=self.interval)
        data = data.drop(["Adj Close", "Volume"], axis = 1)
        return data
