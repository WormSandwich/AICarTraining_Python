import socket
import sys
from random import randint
from struct import *
import stateMessage_pb2

#set up actions
actions = ["FD","BK","RT","LT","ST"]

# create socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# bind socket to port
server_address = ('localhost', 10000)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)

# listen for incoming connections
sock.listen(1)

while True:
    # Wait for a connection
    print('Waiting for connection')
    connection, client_address = sock.accept()
    try:
        print('connection from', client_address)

        # Recieve the data in small chunks and reTransmit it
        while True:
            data = connection.recv(70)
            
            sendData = actions[randint(0,4)]
            sendData = sendData.encode('utf-8')
            #print('recieved {!r}'.format(data))

            print('recieved: ')
            print(data)

            try:
                gameState = stateMessage_pb2.GameState()
                gameState.ParseFromString(data)
                print('Parsed data: ')
                print(gameState.rotX)
            except:
                print('Exception in parsing')

            if data:
                #print('sending data back to the client')
                connection.sendall(sendData)
            else:
                print('no data from', client_address)
                break

    finally:
        connection.close()


