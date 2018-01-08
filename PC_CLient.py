import socket

HOST='192.168.4.1'
#PORT=10000
MESSAGE='[(2,5),(3,4),(10,9)]'
BUFFER_SIZE = 1024

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#s.bind((HOST,PORT))
PORT=1200
s.connect((HOST,PORT))
s.send(MESSAGE)
data=s.recv(BUFFER_SIZE)
while data!=-1:
    print 'received data:',data
    data=s.recv(BUFFER_SIZE)
    


