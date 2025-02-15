import os
import multiprocessing as mp
import psutil

# Test: both should result in 8
print(os.environ["NUMBER_OF_PROCESSORS"])
print(mp.cpu_count())


def spawn():
    procs = list()
    n_cpus = psutil.cpu_count()
    for cpu in range(n_cpus):
        affinity = [cpu]
        d= {'affinity': affinity}
        p = mp.Process(target=run_child, kwargs=d)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        print('joined')

def run_child(affinity):
    proc = psutil.Process()  # get self pid
    print('PID: {pid}'.format(pid=proc.pid))
    aff = proc.cpu_affinity()
    print('Affinity before: {aff}'.format(aff=aff))
    proc.cpu_affinity(affinity)
    aff = proc.cpu_affinity()
    print('Affinity after: {aff}'.format(aff=aff))

if __name__ == '__main__':
    mp.freeze_support()
    spawn()