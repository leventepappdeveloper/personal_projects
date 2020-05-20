# Running the Code          
Run MAIN.py in the Terminal (Python 3).     

# Input Format    
Ticker: name of ticker you want to backtest (i.e. SPY)      
Start Date: YYYY-MM-DD      
End Date: YYYY-MM-DD      
Candle Interval: Valid Intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]      
Restrictions:      
1m: must be within last 30 days, max 5 days between start and end      
2m: must be within last 60 days      
5m: must be within last 60 days      
15m: must be within last 60 days       
1h: must be within last 730 days    

MA Period: averaging period to be backtested        
Output Path/Filename: e.g. /Users/drake/Desktop/output.xlsx OR output.xlsx (file gets put in Working Directory)       

# Output    
Backtesting the so called Confirmation/Validation Strategy. This program returns a comparison of the performances of moving average strategies based on different moving averages (simple, smoothed, exponential). Strategy:     
LONG:     
Entry: green candlestick opens and closes above moving average line       
Exit: candlestick closes below moving average line     

SHORT:     
Entry: red candlestick opens and closes below moving average line       
Exit: candlestick closes above moving average line   

LONG/SHORT:     
Entry: green candlestick opens and closes above moving average line OR red candlestick opens and closes below moving average line    
Exit: candlestick closes below moving average line OR candlestick closes above moving average line      



