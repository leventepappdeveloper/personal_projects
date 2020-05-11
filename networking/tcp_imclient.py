#
# A messaging client in Python - Spring 2020
#
# Names: Levente Papp (lpaa2016)
#

import sys
import socket
from os import _exit as quit

def main():
    # parse arguments
    if len(sys.argv) != 3:
        print("usage: python3 %s <host> <port>" % sys.argv[0]);
        quit(1)
    host = sys.argv[1]
    port = sys.argv[2]

    # TODO 1: open a socket and connect to server
    sock_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_fd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    address = (host, int(port))

    sock_fd.connect(address)

    # message loop
    while(True):
        # TODO 2: send message and receive response
        input_data = input("Enter message for server: ")
        bytes = input_data.encode()
        sock_fd.send(bytes)
        data = sock_fd.recv(50)
        print("Received from server: " + str(data.decode()))


if __name__ == "__main__":
    main()
