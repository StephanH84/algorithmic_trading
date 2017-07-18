import json

import numpy as np


class ProductSet():
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def produce(self):
        s = []

        for a in self.A:
            for b in self.B:
                s.append([a, b])

        return s


class Map():
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination


class PartialMap(dict):
    pass


def encode_np_array(array):
    try:
        result = json.dumps([i for i in array.flatten()])
    except:
        assert True
        result = None
    return result


def decode_np_array(str_, size=3):
    try:
        result = np.asarray(json.loads(str_)).reshape([size, size])
    except:
        assert True
        result = None
    return result