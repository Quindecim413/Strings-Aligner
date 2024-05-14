from subprocess import call
import threading, queue
import numpy as np

from collections import defaultdict
import traceback


# print(lines)

def _worker(q: queue.Queue, thread_jobs, executor):
    while True:
        data = q.get(block=True)
        if data is None:
            q.put(None)
            return

        fn, callback, job_ind, args, kwargs = data
        
        # print('setting thread_job', threading.get_ident(), ' -> ', job_ind)
        thread_jobs[threading.get_ident()] = job_ind
        bad = False
        try:
            res = fn(*args, **kwargs)
        except Exception as e:
            bad = True
            print('error', e)
            print(traceback.format_exc())
            raise e
        finally:
            if bad:
                # executor.shutdown()
                executor._shutdown = True
                del thread_jobs[threading.get_ident()]
                return
        
        if callback:
            callback(job_ind, res)
        q.task_done()
        del thread_jobs[threading.get_ident()]

        if executor._shutdown:
            q.put(None)
            return

class ConcurrentJobsExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers
        
        self.works_queue = queue.Queue()

        self.threads: list[threading.Thread] = []
        
        self._shutdown = False
        self.counters = defaultdict(int)
        self._jobs_callbacks = {}
        self.thread_jobs = {}
        self.pause_jobs_enquing_lock = threading.Lock()
        self.done_jobs_event = threading.Event()
        self.counters_lock = threading.Lock()
        self.max_job_ind = -1

    def start(self):
        self.works_queue = queue.Queue()
        for _ in range(self.max_workers):
            t = threading.Thread(target=_worker, args=(self.works_queue, self.thread_jobs, self))
            t.daemon = True
            t.start()
            self.threads.append(t)

    def shutdown(self):
        print('shuting down')
        self._shutdown = True
        self.works_queue.put(None)
        for thread in self.threads:
            thread.join()
        print('shutdown complete')

    def __enter__(self):
        self.start()
        self.done_jobs_event.set()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print('ConcurrentJobsExecutor.__exit__')
        self.wait_jobs_done()
        self.shutdown()
        if exc_value:
            raise exc_value

    def execute_subtask(self, fn, callback, *args, **kwargs):
        def callback_with_job_ind_and_counter(job_ind, result, callback=callback):
            # print('subtask counters callback 1', self.counters)
            if callback:
                callback(result)
            # print('subtask counters callback 2', self.counters)
            with self.counters_lock:
                self.counters[job_ind] -= 1

            self._check_job_done(job_ind)


        job_ind = self.thread_jobs[threading.get_ident()]
        # print('del thread_job', threading.get_ident(), ':', job_ind)
        # print('subtask counters', self.counters)
        # del self.thread_jobs[threading.get_ident()]
        with self.counters_lock:
            self.counters[job_ind] += 1
        self.works_queue.put((fn, callback_with_job_ind_and_counter, job_ind, args, kwargs))

    def _check_job_done(self, job_ind):
        if self.counters[job_ind] == 0:
            del self.counters[job_ind]
            self._jobs_callbacks[job_ind]()
            del self._jobs_callbacks[job_ind]

        with self.counters_lock:
            if len(self.counters) < self.max_workers:
                if self.pause_jobs_enquing_lock.locked():
                    self.pause_jobs_enquing_lock.release()
            if len(self.counters) == 0:
                if not self.done_jobs_event.isSet():
                    self.done_jobs_event.set()

    def create_job(self, job, callback):
        self.done_jobs_event.clear()

        with self.counters_lock:
            self.max_job_ind += 1
            job_ind = self.max_job_ind
        
        # print('NUM JOBS ENQUED', len(self.counters), self.max_workers)
        if len(self.counters) >= self.max_workers:
            # print('acquire lock')
            self.pause_jobs_enquing_lock.acquire()

        self.thread_jobs[threading.get_ident()] = job_ind
        # print(self.thread_jobs)
        self._jobs_callbacks[job_ind] = callback
        self.execute_subtask(job, None)
    
    def wait_jobs_done(self):
        self.done_jobs_event.wait()


    # def create_job_from_generator(self, generator_obj_callback):
    #     for job_ind, (job, callback) in enumerate(generator_obj_callback):
    #         if len(self.counter) >= self.max_workers:
    #             self.pause_jobs_enquing_event.wait()
            
    #         self.thread_jobs[threading.get_ident()] = job_ind
    #         self._jobs_callbacks[job_ind] = callback
    #         self.execute_subtask(job, None)
