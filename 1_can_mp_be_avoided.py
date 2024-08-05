from time import time
import numpy as np
from multiprocessing.pool import ThreadPool
import os

# * Explanation
# Conventional wisdom to speedup Python apps is broadbly summarized by these two guidelines
# 1. If bottleneck = I/O or network or any other non-CPU intensive task, then use multithreading (threads within the same parent process)
# 2. If bottleneck = CPU, then use multiprocessing (a new process is created, with copy of the parent's memory and data)
# HOWEVER, there are situations when CPU-intensive compute can be accelerated with threads ... and this is always preferred over
# multiprocessing because threading is much simpler to implement, to combine results from different threads etc.
# It's true that generic Python code won't parallelize when using multiple threads because general Python and most libraries retain the 
# global interpreter lock (GIL). While any single thread has the GIL, no other threads can be created and/or executed. However, some
# libraries actually release the GIL for many (but not all) of their operations. Most notably, numpy is one such library.
# This means that CPU-intensive pure numpy code can be accelerated with multithreading. In these instances, the new threads are 
# created on a different CPU core ... that is why we see a speed-up (if they were on the same CPU core, then obviously no speedup
# would be achieved). This can be seen by watching the system monitor when running this following code: only 1 core will 
# go to 100% utilization during the first part of the code, during the second part of the code that invokes ThreadPool, 
# 4 CPU cores will be seen going to 100%. 

arr = np.ones((2048, 1024, 1024), dtype=np.uint8)
expected_sum = arr.sum()

start = time()
for i in range(8):
    arr.sum()
print(f"Sequential: {time() - start:.3g} secs")
start = time()

with ThreadPool(os.cpu_count() - 1) as pool:
    result = pool.map(np.sum, [arr] * 8)
    assert result == [expected_sum] * 8
print(f"Multithreaded: {time() - start:.3g} secs")