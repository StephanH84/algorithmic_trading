import multiprocessing as mp
import psutil
from multiprocess.zmq.src.push_pull.tasksink import sink, bind as sink_bind
from multiprocess.zmq.src.push_pull.taskwork import work, bind as work_bind

def spawn():
    procs = list()

    sink_port = sink_bind()
    sink_p = mp.Process(target=sink)
    sink_p.start()
    procs.append(sink_p)

    #for i in range(1):
    port_b = work_bind(sink_port)
    p = mp.Process(target=work)
    p.start()
    procs.append(p)

    print("port_b: %s" % port_b)
    sink_p.join()


if __name__ == '__main__':
    mp.freeze_support()
    spawn()