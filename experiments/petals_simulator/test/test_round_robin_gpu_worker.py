import threading
import queue

import numpy as np


class Task:
    def __init__(self, function, args=(), kwargs={}):
        # Store the function and arguments
        self.function = function
        self.args = args
        self.kwargs = kwargs

        # Use an event to signal completion
        # An event can have two states: set and unset (also called signaled and unsignaled). 
        # A thread can wait for an event to be set and another thread can set the event. 
        self.event = threading.Event()
        
        # Store the result of the function and any exception
        self.result = None
        self.exception = None

    # Execute the function and store the result
    # Called by the worker thread
    def execute(self):
        try:
            self.result = self.function(*self.args, **self.kwargs)
        except Exception as e:
            self.exception = e
        finally:
            # In either case, set the event to signal completion
            self.event.set()

    # Wait for the task to complete and return the result
    # Called by the thread that submitted the task
    def wait(self):
        self.event.wait()
        if self.exception:
            raise self.exception
        return self.result


# A worker thread that executes tasks from a queue
def worker(task_queue):
    while True:
        task = task_queue.get()
        if task is None:  # Exit signal
            break
        thread_id = threading.get_ident()
        print(f"Worker thread {thread_id} executing task {task.args}")
        task.execute()


def submit_task(task_queue, function, *args, **kwargs):
    task = Task(function, args, kwargs)
    task_queue.put(task)
    return task


def main():
    # Create a worker thread and the related task queue
    task_queue = queue.Queue()
    worker_thread = threading.Thread(target=worker, args=(task_queue,))
    worker_thread.start()

    # Example: using multiple threads to submit tasks and wait for results
    def test_function(x, y):
        return x * y

    def thread_task(task_queue):
        thread_id = threading.get_ident()
        args = np.random.randint(0, 10, size=2)
        task = submit_task(task_queue, test_function, *args)
        print(f"Thread {thread_id} submitted task {task.args} to queue")
        result = task.wait()
        print(f"Thread {thread_id} got result: {result}")

    # Submit tasks from multiple threads
    threads = [threading.Thread(target=thread_task, args=(task_queue,)) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The main thread signals worker to exit
    task_queue.put(None)
    worker_thread.join()


if __name__ == "__main__":
    main()
