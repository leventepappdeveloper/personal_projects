#
# A messaging server in Python - Spring 2020
#
# Names: Levente Papp (lpaa2016)
#

import sys
import socket
from os import _exit as quit

def main():
    # parse arguments
    if len(sys.argv) != 2:
        print("usage: python3 %s <port>" % sys.argv[0])
        quit(1)
    port = sys.argv[1]
    
    # TODO 1: open a socket, bind, and listen
    # OPEN SOCKET
    sock_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_fd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # BIND SOCKET
    address = ("", int(port))
    sock_fd.bind(address)

    # LISTEN FOR CONNECTION
    sock_fd.listen(5)

    # TODO 2: accept connections from client

    # ACCEPT CONNECTION
    conn, addr = sock_fd.accept()


    # message loop
    while(True):
        # TODO 3: receive a message from client and send response
        data = conn.recv(50)
        print("Received from client: " + str(data.decode()))
        input_data = input("Enter message for client: ")
        bytes = input_data.encode()
        conn.send(bytes)

if __name__ == "__main__":
    main()
