import concurrent.futures
import threading, queue
import numpy as np

def _worker(q: queue.Queue, executor):
    while True:
        data = q.get(block=True)
        if data is None:
            q.put(None)
            return
        
        lbl, el, callback, args, kwargs = data
        # print(lbl)
        res = el(*args, **kwargs)
        if callback:
            callback(res)
        q.task_done()
        executor._decrease_counter()
        executor.do_check()

        if executor._shutdown:
            q.put(None)
            return

class ConcurrentExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

        self.works_queue = queue.Queue()

        self.threads: list[threading.Thread] = []
        
        self.donecallback = None
        self.counter = 0
        self.results_tree = []
        self._shutdown = False
        self.res_a_b = None

    def start(self):
        self.works_queue = queue.Queue()

        for _ in range(self.max_workers):
            t = threading.Thread(target=_worker, args=(self.works_queue, self))
            t.daemon = True
            t.start()
            self.threads.append(t)

    def shutdown(self):
        self._shutdown = True
        self.works_queue.put(None)

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.wait_till_end()
        self.shutdown()
        self.works_queue = queue.Queue()
        if exc_value:
            raise exc_value

    def create_job(self, lbl, fn, callback, *args, **kwargs):
        self._increase_counter()
        self.works_queue.put((lbl, fn, callback, args, kwargs))
    
    def _increase_counter(self):
        self.counter += 1

    def _decrease_counter(self):
        self.counter -= 1
    
    def wait_till_end(self):
        for thread in self.threads:
            thread.join()

    def do_check(self):
        if self.counter == 0:
            self.shutdown()
            