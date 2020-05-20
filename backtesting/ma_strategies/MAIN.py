import tkinter as tk
"""
Author: Levente Papp
Date: 5/20/2020
"""
import Tester as tr
import pandas as pd

class UserInterface():
    """
    Class that sets up the user interfacea and runs the backtesting algorithms.
    """
    def __init__(self):
        pass

    def run(self):

        def process_input():
            ticker = e1.get()
            start_date = e2.get()
            end_date = e3.get()
            candle_interval = e4.get()
            ma_period = int(e5.get())
            path = e6.get()

            df = tr.MAIN_Tester(ticker, start_date, end_date, candle_interval, ma_period).run()

            writer = pd.ExcelWriter(path)
            sheet_name = ticker + " " + start_date + " " + end_date
            df.to_excel(writer, sheet_name)
            writer.save()


        master = tk.Tk()
        master.title("MA Strategy Backtester")

        tk.Label(master, text="Ticker").grid(row=0)
        tk.Label(master, text="Start Date").grid(row=1)
        tk.Label(master, text="End Date").grid(row=2)
        tk.Label(master, text="Candle Interval").grid(row=3)
        tk.Label(master, text="MA Period").grid(row=4)
        tk.Label(master, text="Output Path/Filename").grid(row=5)

        e1 = tk.Entry(master)
        e2 = tk.Entry(master)
        e3 = tk.Entry(master)
        e4 = tk.Entry(master)
        e5 = tk.Entry(master)
        e6 = tk.Entry(master)

        e1.grid(row=0, column=1)
        e2.grid(row=1, column=1)
        e3.grid(row=2, column=1)
        e4.grid(row=3, column=1)
        e5.grid(row=4, column=1)
        e6.grid(row=5, column=1)

        tk.Button(master, text='Run', command=process_input).grid(row=6, column=0, pady=4)
        tk.Button(master, text='Quit', command=master.quit).grid(row=6, column=1, pady=4)

        master.mainloop()

if __name__ == "__main__":
    ui = UserInterface()
    ui.run()

