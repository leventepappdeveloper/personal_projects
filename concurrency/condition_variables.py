# LEVENTE PAPP, 4/25/2020, CS105 Pomona College

from __future__ import with_statement
from threading import Thread, Lock, Condition, Semaphore
from os import _exit as quit
import time, random

####################################################### 
#                      
# Partner 1:
#
# Partner 2:
# 
#######################################################


# Add locks and condition variables to ensure that
#    (a) the club is exclusively Goth or Hipster, i.e. no Goth
#        should enter as long as there are Hipsters in the club,
#        and vice versa,
#    (b) the club should always be used as long as there are 
#        customers
#    Note that starvation is not something you need to worry 
#    about. If the club becomes Goth and remains exclusively 
#    Goth for all time, the waiting Hipsters will just have
#    to get old at the door. 
#
# Modify only the code of the class Club to make the program
# correct.
# Place your synchronization variables inside the Club instance.
# Make sure nobody is holding a Club synchronization variable
# while executing outside the Club code.



def hangout():
    time.sleep(random.randint(0, 2))


class Club:
    def __init__(self, capacity):
        self.goth_count = 0               # num goths in club
        self.hipster_count = 0            # num hipsters in club
        self.capacity = capacity          # only used for optional questions

        self.lock = Lock()
        self.club = Condition(self.lock)


    def __sanitycheck(self):
        if self.goth_count > 0 and self.hipster_count > 0:
            print("sync error: bad social mixup! Goths = %d, Hipsters = %d" %  (self.goth_count, self.hipster_count))
            quit(1)
        if self.goth_count>self.capacity or self.hipster_count>self.capacity:
            print("sync error: too many people in the club! Goths = %d, Hipsters = %d" %  (self.goth_count, self.hipster_count))
            quit(1)
        if self.goth_count < 0 or self.hipster_count < 0:
            print("sync error: lost track of people! Goths = %d, Hipsters = %d" %  (self.goth_count, self.hipster_count))
            quit(1)

        
    def goth_enter(self):

        self.lock.acquire()

        while(self.hipster_count > 0):
            self.club.wait()

        self.goth_count +=1               
        self.__sanitycheck()

        self.lock.release()


    def goth_exit(self):

        self.lock.acquire()

        self.goth_count -= 1
        self.__sanitycheck()

        if self.goth_count == 0:
            self.club.notify()

        self.lock.release()


    def hipster_enter(self):

        self.lock.acquire()

        while (self.goth_count > 0):
            self.club.wait()

        self.hipster_count += 1
        self.__sanitycheck()

        self.lock.release()

        
    def hipster_exit(self):

        self.lock.acquire()

        self.hipster_count -= 1
        self.__sanitycheck()

        if self.hipster_count == 0:
            self.club.notify()

        self.lock.release()


class Goth(Thread):
    def __init__(self, id):
        Thread.__init__(self)
        self.id = id

    def run(self):
        global daclub

        while True:
            print("goth #%d: wants to enter" % self.id)
            daclub.goth_enter()
            print("goth #%d: in the club" % self.id)
            print("goths in club: %d" % daclub.goth_count)
            hangout()
            daclub.goth_exit()
            print("goth #%d: left club" % self.id)
            print("goths in club: %d" % daclub.goth_count)
            
class Hipster(Thread):
    def __init__(self, id):
        Thread.__init__(self)
        self.id = id

    def run(self):
        global daclub

        while True:
            print("hipster #%d: wants to enter" % self.id)
            daclub.hipster_enter()
            print("hipster #%d: in the club" % self.id)
            print("hipsters in club: %d" % daclub.hipster_count)
            hangout()
            daclub.hipster_exit()
            print("hipster #%d: left club" % self.id)
            print("hipsters in club: %d" % daclub.hipster_count)


NUMGOTH = 2
NUMHIPSTER = 2
CAPACITY = NUMGOTH + NUMHIPSTER
daclub = Club(CAPACITY)


def main():
    for i in range(0, NUMGOTH):
        g = Goth(i)
        g.start()    
    for i in range(0, NUMHIPSTER):
        h = Hipster(i)
        h.start()    

if __name__ == "__main__":
    main()
