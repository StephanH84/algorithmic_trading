import socket
import random
import time
import json
import psutil

def thread(cls, host, port, affinity):
    print("Start thread for port: %s, host: %s" % (host, port))
    proc = psutil.Process()  # get self pid
    proc.cpu_affinity([affinity])
    aff = proc.cpu_affinity()
    print('Affinity after: {aff}'.format(aff=aff))

    cls.initialize_sock_conn(host, port)
    cls.event_loop()


class Worker():
    @classmethod
    def __del__(cls):
        try:
            cls.socket.close()
        except:
            pass

    @classmethod
    def initialize_sock_conn(cls, host, port):
        print("initialize_sock_conn")
        cls.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cls.socket.connect((host, port))
        print("Connected to port %s" % port)

    @classmethod
    def event_loop(cls):
        print("event loop")
        # Currently for data channel
        while True:
            # PULL section
            cls.send_data("PULL")
            OK_received = False
            while not OK_received:
                fromServer = cls.socket.recv(10)
                if fromServer == b'OK':
                    print("OK(PULL) received")
                    OK_received = True
                    data_received = cls.socket.recv(1024)
                    print("data_received: %s" % data_received)

            # Computational section, modelled as time.sleep(10)
            time.sleep(2)
            data = cls.generate_data()

            # PUSH section
            cls.send_data("PUSH")
            OK_received = False
            while not OK_received:
                fromServer = cls.socket.recv(10)
                if fromServer == b'OK':
                    print("OK(PUSH) received")
                    OK_received = True
                    cls.send_data(data)

    @staticmethod
    def generate_data():
        return json.dumps([random.randint(0, 10) for n in range(1)])

    @classmethod
    def send_data(cls, data):
        cls.socket.sendall(bytes(data.encode('utf8')))
        print("Data sent")
        # data = cls.socket.recv(1024)
        # print('Received', repr(data))