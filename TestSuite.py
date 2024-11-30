
import multiprocessing as mp
import time


from functools import partial

import os

# Limit the number of threads for various linear algebra libraries
os.environ['OMP_NUM_THREADS'] = '1'          # OpenMP
os.environ['OPENBLAS_NUM_THREADS'] = '1'     # OpenBLAS
os.environ['MKL_NUM_THREADS'] = '1'          # Intel MKL
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'   # Apple Accelerate/vecLib
os.environ['NUMEXPR_NUM_THREADS'] = '1'      # NumExpr

class TestSuite:
    def __init__(self, cases):
        self.cases = cases  # how many cases
        self.tests = []
        self.max_parallel = mp.cpu_count()-1
        self.started = 0
        self.ended = 0

    def add_test(self, test, case):
        test.case = case  # Assign the case number to the test
        self.tests.append(test)

    def run_test(self, test):
        log = test.runTest(test.noise, test.period)
        return test.case, log

    def start_tests(self):
        with mp.Pool(processes=self.max_parallel) as pool:
            results = pool.map(self.run_test, self.tests)
        self.results = results

    def gather_data(self):
        data = [[] for _ in range(self.cases)]
        for case, log in self.results:
            if log.step > 0:
                data[case].append(log)
        return self.results

    def reset(self):
        self.tests = []
        self.started = 0
        self.ended = 0

class TestSuitee:
    def __init__(self, cases):
        self.cases = cases # how many cases
        self.tests = []
        self.max_parallel = mp.cpu_count()
        self.started = 0
        self.ended = 0
        self.queues = [mp.Manager().Queue() for n in range(self.cases)]

    def add_test(self, test, case):
        test.add_queue(self.queues[case])
        self.tests.append(test)

    def selective_join(self, timeout):
        for test in self.running:
            test.join(timeout)
            if not test.is_alive():
                self.running.remove(test)
                self.ended += 1

    def start_tests(self):
        self.start_pool()
        #for test in self.tests:
            #test.start()

        #for test in self.tests:
            #test.join()
        
    def start_pool(self):
        test_count = len(self.tests)
        self.running = []
        while self.ended < test_count:
            while ((self.started - self.ended) < self.max_parallel) and (self.started < test_count):
                self.tests[self.started].start()
                self.running.append(self.tests[self.started])
                self.started += 1

            time.sleep(0.5) 
            #self.selective_join(0.1)
            #print(self.ended)
            self.ended = self.get_ended_count()

    def get_ended_count(self):
        size = 0
        for n in range(self.cases):
            q = self.queues[n]
            size += q.qsize()

        return size

    def join_tests(self, timeout=0):
        for test in self.tests:
            test.join(timeout)

    def reset(self):
        self.tests = []
        self.queues = [mp.Manager().Queue() for n in range(self.cases)]
        self.started = 0
        self.ended = 0

    def gather_data(self):
        data = []
        for n in range(self.cases):
            logs = []
            q = self.queues[n]
            while not q.empty():
                log = q.get()
                if log.step > 0:
                    logs.append(log)
            data.append(logs)

        return data

    def kill_all(self):
        for test in self.tests:
            if test.is_alive():
                test.terminate()
                test.join()




