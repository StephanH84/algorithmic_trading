import socket
from multiprocess.socket.src.asynch.worker import Worker, thread as worker_thread
import multiprocessing as mp
import psutil
import time
import json

def encode_bytes(txt):
    return bytes(txt.encode('utf8'))

def decode_bytes(byt):
    return str(byt.decode('utf8'))

class GlobalState():
    BEGINNING = 1

class GlobalProcess():

    def __init__(self):
        self.internal_state = GlobalState.BEGINNING

        self.initialize(3)

    def run(self):
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
                time.sleep(0.3)
                try:
                    data = conn.recv(128)
                except:
                    data = None
                    print("No data was received from %s" % addr[1])

                if data:
                    print("from: %s -- Data received Received: %s" % (addr[1], data))

                if data == b'PUSH': # PUSH COMMAND RECEIVED: getting data
                    print("PUSH received")
                    conn.sendall(b'OK')
                    t0 = time.time()
                    while True:
                        try:
                            data = conn.recv(2048)
                            break
                        except BlockingIOError:
                            t1 = time.time()
                            if t1 - t0 > 1:
                                data = None
                                break
                            pass

                    if data is not None:
                        data_decode = decode_bytes(data)
                        self.data_storage.append([addr[1], data_decode])

                elif data == b'PULL':
                    print("PULL received")
                    conn.sendall(b'OK')
                    conn.sendall(encode_bytes(json.dumps(self.data_storage)))


    def initialize(self, N):
        self.data_storage = []
        self.procs = []
        port = 11113

        n_cpus = psutil.cpu_count()
        N = min([n_cpus, N])

        for n in range(N):
            # initialize server socket

            # set up connections server connection

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.0)

            # https://stackoverflow.com/questions/1365265/on-localhost-how-to-pick-a-free-port-number
            HOST = ''  # Symbolic name meaning all available interfaces
            PORT = port  # Arbitrary non-privileged port
            sock.bind((HOST, PORT))

            host, port = "localhost", port

            # start a worker thread with parameter the socket port
            w = Worker()
            d = {'host': host, 'port': port, 'cls': w.__class__, 'affinity': n}
            p = mp.Process(target=worker_thread, kwargs=d)
            p.start()

            sock.listen(1)

            while True:
                try:
                    conn, addr = sock.accept()
                    break
                except (BlockingIOError, socket.timeout):
                    pass

            self.procs.append([sock, conn, addr])

            port += 1

def main():
    g = GlobalProcess()
    g.run()


if __name__ == '__main__':
    mp.freeze_support()
    main()