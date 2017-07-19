import socket
import random
import time

class Worker():
    @classmethod
    def thread(cls, port):
        cls.initialize_sock_conn(port)
        cls.event_loop()

    @classmethod
    def __del__(cls):
        try:
            Worker.socket.close()
        except:
            pass

    @classmethod
    def initialize_sock_conn(cls, port):
        cls.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cls.socket.connect(('localhost', port))

    @classmethod
    def event_loop(cls):
        # Currently for data channel
        while True:
            time.sleep(10)
            data = cls.generate_data()
            Worker.send_data(data)

    @staticmethod
    def generate_data():
        return [random.randint(0,1) for n in range(100)]

    @classmethod
    def send_data(cls):
        cls.socket.sendall(b'Hello, world')
        data = cls.socket.recv(1024)
        print('Received', repr(data))