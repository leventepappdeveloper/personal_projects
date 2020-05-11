'''
Author: Levente Papp
This program displays a simple use case of synchronizing threads using locks in Pyhton.
'''
import threading
import time
import random

'''
This class creates an unsynchronized "Trader" thread
'''
class UnSynchronized_Trader(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    # Trader thread purchases a random number of shares out of 500
    def run(self):
        global available_unsync
        while available_unsync < 500:
            amt = random.randint(0, 500 - available_unsync)
            time.sleep(0.2)
            available_unsync += random.randint(0, amt)

'''
This class creates an synchronized "Trader" thread.
In this case, a Trader can only make a purchase once the previous one has finished
their transaction, thus avoiding a race condition. 
'''
class Synchronized_Trader(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    # Trader thread purchases a random number of shares out of 500
    # Synchronization implemented through Locks
    def run(self):
        global available_sync
        lock.acquire()
        while available_sync < 500:
            amt = random.randint(0, 500 - available_sync)
            time.sleep(0.2)
            available_sync += random.randint(0, amt)
        lock.release()

available_unsync = 0
available_sync = 0
lock = threading.Lock()

# Main Thread creates 5 unsynchronized and 5 synchronized Traders to display the need for locks
if __name__ == '__main__':

    for i in range(5):
        trader = UnSynchronized_Trader()
        trader.start()
    for j in range(5):
        trader.join()

    # Problem arises here: more shares will get purchased than there are available.
    print("Unsynchronized traders bought " + str(available_unsync) + " out of 500 available shares.")

    for i in range(5):
        trader = Synchronized_Trader()
        trader.start()

    for j in range(5):
        trader.join()

    print("Synchronized traders bought " + str(available_sync) + " out of 500 available shares.")