def encode_bytes(txt):
    return bytes(txt.encode('utf8'))


def decode_bytes(byt):
    return str(byt.decode('utf8'))


MAX_SIZE = 20