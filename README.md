# File Descriptions      

## synchronized_traders.py       
This simple example displays a use case for synchronized locks in Python. The code creates two kinds of traders: 
synchronized and unsynchronized. Five traders are trying to purchase a total of 500 shares, all at once. 
While unsynchronized traders end up purchasing more than 500 shares (due to an existing race condition on the global 
variable), synchronized traders will end up purchasing exactly 500 shares. 
