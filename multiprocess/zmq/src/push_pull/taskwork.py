# Task worker
# Connects PULL socket to tcp://localhost:5557
# Collects workloads from ventilator via that socket
# Connects PUSH socket to tcp://localhost:5558
# Sends results to sink via that socket
#
# Author: Lev Givon <lev(at)columbia(dot)edu>

import sys
import time
import zmq


context = zmq.Context()


# Socket to receive messages on
receiver = context.socket(zmq.PULL)

# Socket to send messages to
sender = context.socket(zmq.PUSH)

def bind(port_a):
    sender.connect("tcp://localhost:%s" % port_a)

    success = False
    port = port_a + 1
    while not success:
        try:
            receiver.bind("tcp://*:%s" % port)
            print("Successfully bound as PULL to port: %s" % port)
            success = True
        except zmq.error.ZMQError:
            port += 1
    return port


def work():
    # Process tasks forever
    while True:
        s = receiver.recv()

        # Simple progress indicator for the viewer
        sys.stdout.write('.')
        sys.stdout.flush()

        # Do the work
        time.sleep(int(s)*0.001)

        # Send results to sink
        sender.send(b'')
