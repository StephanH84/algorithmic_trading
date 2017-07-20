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
    def __init__(cls, t_max=5):
        cls.network = Network()
        cls.tradingEnv = TradingEnv()
        cls.t_max = t_max

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
        t = 1
        while True:
            # PULL section
            weights =cls.pull_from_server()

            # computational section
            gradients, local_T = cls.algorithm_loop_body(t, cls.t_max, weights)

            # PUSH section

            gradients_json = json.dumps(gradients)
            data = "GRADIENTS" + gradients
            cls.push_to_server(data)

            local_T_str = str(local_T)
            data = "local_T" + local_T_str
            cls.push_to_server(data)

    @classmethod
    def push_to_server(cls, data_str):
        cls.socket.sendall("PUSH")
        OK_received = False
        while not OK_received:
            fromServer = cls.socket.recv(10)
            if fromServer == b'OK':
                print("OK(PUSH) received")
                OK_received = True
                bytes_to_send = encode_bytes(data_str)
                bytes_to_send_size = encode_bytes(str(len(bytes_to_send)))
                bytes_to_send_size = b'0' * (MAX_SIZE - len(bytes_to_send_size)) + bytes_to_send_size
                cls.socket.sendall(bytes_to_send_size)
                cls.socket.sendall(bytes_to_send)
                print("data sent: %s" % data_str)

    @classmethod
    def pull_from_server(cls):
        cls.socket.sendall("PULL")
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

    @classmethod
    def computational_section(cls):
        data = cls.generate_data()
        return data

    @staticmethod
    def generate_data():
        return json.dumps([random.randint(0, 10) for n in range(1)])

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
    def algorithm_loop_body(cls, t, t_max, weights):
        local_T = 0
        t_start = t # TODO: initialize to 1

        cls.network.update_weights(weights)

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

        return gradients, local_T
