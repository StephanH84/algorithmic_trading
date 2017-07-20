import json
import multiprocessing as mp
import socket
import time

import psutil

from asynch.common import encode_bytes, decode_bytes, MAX_SIZE
from AT.asynchronous_actor_critique.src.agent import Agent, thread as worker_thread
import numpy as np


class GlobalState():
    BEGINNING = 1

class ParameterServer():
    # The parameter server and spawning all other threads
    def __init__(self, alpha, eta):
        self.alpha = alpha
        self.eta = eta
        self.internal_state = GlobalState.BEGINNING

        self.weights_common = np.array()
        self.weights_policy = np.array()
        self.weights_value = np.array()

        self.g_common = np.array()
        self.g_policy = np.array()
        self.g_value = np.array()

        self.initialize(2)

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
                    content_length = self.receive(conn, 20)
                    content_length_int = int(content_length)
                    data = self.receive(conn, content_length_int)

                    if data is not None:
                        data_decode = decode_bytes(data)
                        self.data_storage.append([addr[1], data_decode])

                    conn.sendall(b'PUSH_END')

                elif data == b'PULL':
                    print("PULL received")
                    bytes_to_send = encode_bytes(json.dumps(self.data_storage))
                    bytes_to_send_size = encode_bytes(str(len(bytes_to_send)))
                    bytes_to_send_size = b'0' * (MAX_SIZE - len(bytes_to_send_size)) + bytes_to_send_size
                    conn.sendall(b'OK' + bytes_to_send_size)
                    conn.sendall(bytes_to_send)

    def receive(self, conn, content_length_int):
        t0 = time.time()
        while True:
            try:
                data = conn.recv(content_length_int)
                break
            except BlockingIOError:
                t1 = time.time()
                if t1 - t0 > 1:
                    data = None
                    break
                pass
        return data

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
            w = Agent()
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

    def initialize_weights(self, shapes):
        self.weights_common = np.array()
        self.weights_policy = np.array()
        self.weights_value = np.array()

        self.g_common = np.array()
        self.g_policy = np.array()
        self.g_value = np.array()


    def update_weights(self, gradients, eps=1e-6):
        # Perform stochastic gradient descent
        # e.g. RMSProb of gradients with shared statistics
        policy_gradients, value_gradients, common_gradients = gradients

        self.g_policy = self.alpha * self.g_policy + (1 - self.alpha) * policy_gradients**2
        self.weights_policy -= self.eta * policy_gradients / np.sqrt(self.g_policy + eps)

        self.g_policy = self.alpha * self.g_policy + (1 - self.alpha) * policy_gradients**2
        self.weights_policy -= self.eta * policy_gradients / np.sqrt(self.g_policy + eps)


        self.g_policy = self.alpha * self.g_policy + (1 - self.alpha) * policy_gradients**2
        self.weights_policy -= self.eta * policy_gradients / np.sqrt(self.g_policy + eps)


def main():
    g = ParameterServer()
    g.run()


if __name__ == '__main__':
    mp.freeze_support()
    main()