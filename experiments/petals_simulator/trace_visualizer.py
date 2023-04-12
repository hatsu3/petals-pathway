import time
import os
import threading
import json

class TraceVisualizer:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            ts = int(time.time() * 1e6)
            func_entry = {
                'name': func.__name__,
                'cat': 'function',
                'ph': 'B',
                'ts': ts,
                'pid': os.getpid(),
                'tid': threading.get_ident() % 645713,
                'args': {}
            }
            with open(self.log_file_path, 'a') as f:
                f.write(f"{json.dumps(func_entry)},\n")

            result = func(*args, **kwargs)

            ts = int(time.time() * 1e6)
            func_exit = {
                'name': func.__name__,
                'cat': 'function',
                'ph': 'E',
                'ts': ts,
                'pid': os.getpid(),
                'tid': threading.get_ident() % 645713,
                'args': {}
            }
            with open(self.log_file_path, 'a') as f:
                f.write(f"{json.dumps(func_exit)},\n")
            
            return result
        
        return wrapper

'''
Usage of TraceVisualizer as a decorator:

@TraceVisualizer(log_file_path='trace.json')
def f():
    # function body

'''
