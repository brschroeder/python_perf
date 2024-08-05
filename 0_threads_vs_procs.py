from time import time
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import numpy as np

# * Explanation
# All threads running within one process share memory. But each process has its own memory space.
# When a new process is created, the parent process creatse a copy of it's memory and sends the child process whatever
# it has stored in memory, which in this example code is a 1024x1024x1024 matrix.
# The parent process first serializes this data with pickle, then sends it to child process. The child process
# then unpickles this data before it can be used. If data volumes are "big", then this pickling/unpickling 
# can add a significant amount of time. This is always a trade-off: if the computation times >> serializing/deserializing
# times, then this is an acceptable trade-off.
# This example performs the same small calculation under three different conditions
# 1. In the parent process, using the same thread as the main() program i.e. this is the most common form of Python programming
# 2. In a new thread (with same parent process), here memory is shared between the orginal thread of main() and the new thread we create
# 3. In a new process, here the parent process needs to serialize data, send it, then child process needs to deserialize it
# Since this is a small calculation, the overhead of (de)serializing dominates and "new process" variant is slowest

def main():
    arr = np.ones((10, 10, 10), dtype=np.uint8)
    start = time()
    expected_sum = np.sum(arr)
    print(f"Parent process and thread: {time() - start:.3g}")

    with ThreadPool(1) as threadpool:
        start = time()
        assert(threadpool.apply(np.sum, (arr,)) == expected_sum)
        print(f"New thread pool: {time() - start:.3g}")
    
    with mp.get_context("spawn").Pool(1) as processpool:
        start = time()
        assert(processpool.apply(np.sum, (arr,)) == expected_sum)
        print(f"New process pool: {time() - start:.3g}")

if __name__ == "__main__":
    main()