import socket

HOST='localhost'
PORT=int( raw_input("Enter the port no. to use : "))

BUFFER=1034

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind((HOST,PORT))
s.listen(1)
print "Binding complete... Listening for connection..."

while (1):
    #print "here"
    conn,addr = s.accept()
    print 'Conn Addr:',addr
    data = conn.recv(BUFFER)
    if not data:
        break
    print 'rec data:',data
    #conn.send(data)

conn.close()
