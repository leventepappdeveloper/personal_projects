import robin_stocks as rh
import json
import pandas as pd
import schedule
import time
import datetime
import xlsxwriter

'''
Author: Levente Papp
This program allows the user to query and store real-time options bid/ask
price data from the Robinhood API. 
'''
def log_in():
    # NOTE: need to update this line to point to log-in info
    path_to_config = "/Users/levente/Desktop/Robinhood/config.json"
    content = open(path_to_config).read()
    config = json.loads(content)
    rh.login(config['username'], config['password'])

def initialize_excel_file(filename, option_contracts):
    workbook = xlsxwriter.Workbook(filename)
    workbook.add_worksheet()
    workbook.close()
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    empty_data_frame = pd.DataFrame(columns=["Time", "bid_price", "ask_price", "stock_price"])
    file = open("option_contracts", "r")
    lines = file.readlines()
    for line in lines:
        option_contracts.append(line)
        empty_data_frame.to_excel(writer, sheet_name=line, index=False)
    writer.save()

def get_option_spread(symbol, expirationDate, strike, type):
    data = rh.find_options_for_stock_by_expiration_and_strike(symbol, expirationDate, strike)
    price = rh.get_latest_price(symbol)
    if type == "put":
        res = data[0]
    else:
        res = data[1]
    now = datetime.datetime.now()
    time = now.strftime("%y-%m-%d %H:%M:%S")
    data = [{'Time': time, 'bid_price': res["bid_price"], 'ask_price': res["ask_price"], 'stock_price': price[0]}]
    df = pd.DataFrame(data)
    return df

def update_excel_file(filename, option_contracts):
    sheets = pd.read_excel(filename, sheet_name=option_contracts, header=0)
    sheet_names = sheets.keys()
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    for sheet_name in sheet_names:
        split_option = sheet_name.split(" ")
        symbol = split_option[0]
        strike = split_option[1]
        expiration_date = split_option[2]
        call_or_put = split_option[3]
        df = sheets[sheet_name]
        df.columns = ["Time", "bid_price", "ask_price", "stock_price"]
        df2 = get_option_spread(symbol, expiration_date, strike, call_or_put)
        df = df.append(df2)
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()

def start(filename, option_contracts):
    initialize_excel_file(filename, option_contracts)
    schedule.every().minute.do(update_excel_file, filename, option_contracts)

if __name__ == "__main__":
    log_in()

    date = datetime.datetime.now().strftime("%y-%m-%d")
    filename = str(date) + ".xlsx"
    option_contracts = []

    startTime = "10:44"
    endTime = "13:00"
    schedule.every().day.at(startTime).do(start, filename, option_contracts)

    while datetime.datetime.now().strftime("%H:%M") != endTime:
        schedule.run_pending()
        time.sleep(1)


