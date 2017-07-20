# WORKER thread thus in effect the AGENT
import json
import random
import socket
import time
from network import Network
from trading_env import TradingEnv

import psutil

from asynch.common import encode_bytes, MAX_SIZE

class EndOfData(Exception):
    pass


def thread(cls, host, port, affinity):
    print("Start thread for port: %s, host: %s" % (host, port))
    proc = psutil.Process()  # get self pid
    proc.cpu_affinity([affinity])
    aff = proc.cpu_affinity()
    print('Affinity after: {aff}'.format(aff=aff))

    cls.initialize_sock_conn(host, port)
    cls.event_loop()


class Agent():
    @classmethod
    def __init__(cls):
        cls.network = Network()
        cls.tradingEnv = TradingEnv()

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
                fromServer = bytes(cls.socket.recv(22))
                if fromServer.startswith(b'OK'):
                    print("OK(PULL) received")
                    OK_received = True

                    content_length = fromServer.lstrip(b'OK')
                    print("content_length: %s" % content_length)
                    content_length_int = int(content_length)
                    data_received = cls.socket.recv(content_length_int)
                    print("data_received: %s, length: %s" % (data_received, len(data_received)))

            # Computational section, modelled as time.sleep(10)
            time.sleep(2)
            data = cls.computational_section()

            # PUSH section
            cls.send_data("PUSH")
            OK_received = False
            while not OK_received:
                fromServer = cls.socket.recv(10)
                if fromServer == b'OK':
                    print("OK(PUSH) received")
                    OK_received = True
                    bytes_to_send = encode_bytes(data)
                    bytes_to_send_size = encode_bytes(str(len(bytes_to_send)))
                    bytes_to_send_size = b'0' * (MAX_SIZE - len(bytes_to_send_size)) + bytes_to_send_size
                    cls.socket.sendall(bytes_to_send_size)
                    cls.send_data(data)

    @classmethod
    def computational_section(cls):
        data = cls.generate_data()
        return data

    @staticmethod
    def generate_data():
        return json.dumps([random.randint(0, 10) for n in range(1)])

    @classmethod
    def send_data(cls, data):
        cls.socket.sendall(encode_bytes(data))
        print("Data sent: %s" % data)
        # data = cls.socket.recv(1024)
        # print('Received', repr(data))

    @classmethod
    def pull_weights(cls):
        pass

    @classmethod
    def push_gradients(cls):
        pass

    @classmethod
    def handle_T(cls):
        pass

    @classmethod
    def algorithm_loop_body(cls, t, t_max):
        local_T = 0
        t_start = t # TODO: initialize to 1

        cls.network.update_weights(cls.pull_weights())

        states = {}
        rewards = {}
        actions = {}

        state = cls.tradingEnv.pull_next_state()

        states[t] = state

        wasTerminal = False
        while True:
            action = cls.network.evaluate_policy(state)

            new_state, reward, EOG = cls.tradingEnv.act(action)

            state = new_state
            states[t + 1] = state

            rewards[t] = reward

            actions[t] = action

            if cls.tradingEnv.state_is_terminal(new_state):
                wasTerminal = True
                break

            if (t - t_start == t_max):
                break

            if EOG:
                raise EndOfData()

            t += 1
            local_T += 1


        if wasTerminal:
            R_tp1 = 0
        else:
            R_tp1 = cls.network.evaluate_value(state[t+1])

        R = {}
        for i in range(t, t_start + 1, -1):
            R[i] = rewards[i] + cls.gamma * R[i + 1]


        # calculate accumulated gradients
        state_batch = [state for state in states]
        action_batch = [action for action in actions]
        R_batch = [R_ for R_ in R]

        gradients = cls.network.calc_gradients(states, actions, R)

        cls.push_gradients(gradients)

        cls.handle_T(local_T)