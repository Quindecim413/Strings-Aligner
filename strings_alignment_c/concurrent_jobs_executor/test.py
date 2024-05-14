from threading import Thread
# from . import ConcurrentJobsExecutor
from . import ConcurrentJobsExecutor
import time
import numpy as np

def subtask(ind):
    time.sleep(2)
    return -ind**2

def worker(code, sleep_time, executor: ConcurrentJobsExecutor, do_subtask=False):
    print(code, 'ENTER', sleep_time)
    time.sleep(sleep_time)
    if do_subtask:
        print(code, 'INTO SUBTASK')
        executor.execute_subtask(subtask, lambda res, code=code: print(code, 'subtask result', res), code)
    print(code, 'FINISH')


# with ConcurrentJobsExecutor(2) as executor:
#     for ind, sleep in enumerate(np.arange(0.2, 0.5, 0.05)):
#         executor.create_job(lambda ind=ind, sleep=sleep: worker(ind, sleep, executor, ind%2==0),
#             lambda ind=ind, sleep=sleep: print('DONE', ind, sleep)
#         )
#         print('Enqued', ind)

executor = ConcurrentJobsExecutor(1)
executor.start()
start = time.time()
for ind, sleep in enumerate(range(1, 10), start=1):
    
    executor.create_job(lambda ind=ind, sleep=sleep: worker(ind, sleep, executor, ind%2==0),
        lambda ind=ind, sleep=sleep: print('DONE', ind, sleep)
    )
    print('num of jobs executing', len(executor.counters))
    print('Enqued', ind)
    print(ind)
executor.wait_jobs_done()
end = time.time()
print('elapsed', end-start)
time.sleep(1)
executor.shutdown()