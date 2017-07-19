import socket
from multiprocess.socket.src.asynch.worker import Worker
import multiprocessing as mp

class GlobalState():
    BEGINNING = 1

class GlobalProcess():

    def __init__(self):
        self.internal_state = GlobalState.BEGINNING

        self.initialize(3)
        self.event_loop()

    def __del__(self):
        try:
            for sock, conn, _ in self.procs:
                sock.close()
                conn.close()
        except:
            pass


    def event_loop(self):
        while True:
            for sock, conn, addr in self.procs:
                data = conn.recv(1024)
                if data:
                    print("Received: %s -- from: %s" % (data, addr))

    def initialize(self, N):
        self.procs = []
        port = 50007
        for n in range(N):
            # initialize server socket

            # set up connections server connection

            HOST = ''  # Symbolic name meaning all available interfaces
            PORT = port  # Arbitrary non-privileged port

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            sock.bind((HOST, PORT))


            # start a worker thread with parameter the socket port
            w = Worker()
            p = mp.Process(w.__class__.thread(), kwargs={'port': port, 'cls': w.__class__})
            p.start()


            sock.listen(1)
            conn, addr = sock.accept()


            self.procs.append([sock, conn, addr])

            port += 1
