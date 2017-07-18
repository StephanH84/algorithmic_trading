# Task sink
# Binds PULL socket to tcp://localhost:5558
# Collects results from workers via that socket
#
# Author: Lev Givon <lev(at)columbia(dot)edu>

import sys
import time
import zmq


context = zmq.Context()

# Socket to receive messages on
receiver = context.socket(zmq.PULL)

def bind():
    success = False
    port = 1
    while not success:
        try:
            receiver.bind("tcp://*:%s" % port)
            print("Successfully bound as PULL to port: %s" % port)
            success = True
        except zmq.error.ZMQError:
            port += 1
    return port


def sink():
    # Wait for start of batch
    s = receiver.recv()

    # Start our clock now
    tstart = time.time()

    # Process 100 confirmations
    for task_nbr in range(100):
        s = receiver.recv()
        if task_nbr % 10 == 0:
            sys.stdout.write(':')
        else:
            sys.stdout.write('.')
        sys.stdout.flush()

    # Calculate and report duration of batch
    tend = time.time()
    print("Total elapsed time: %d msec" % ((tend-tstart)*1000))
