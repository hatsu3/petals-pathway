from enum import Enum
import json
import queue
import socket
import threading
import time
from typing import Any

from geopy import Point

from .multitask_model import MultiTaskModel, Stage
from .latency_estimator import LatencyEstimator
from .dht import DistributedHashTable
from .messages import InferRequest, InferResponse
from .stage_profiler import ProfilingResults


def simulated_execution(stage: Stage, batch_size: int, prof_results: ProfilingResults):
    latency = prof_results.get_latency(stage.name, batch_size)
    time.sleep(latency / 1000)


TASK_FUNC_REGISTRY = {
    "simulated_execution": simulated_execution,
}


# Simulates executing a stage of the multi-task model on GPU
class GPUTask:
    def __init__(self, request: InferRequest, func_name: str, args=(), kwargs={}):
        self.request = request

        # Store the function and arguments
        self.func_name = func_name
        self.function = TASK_FUNC_REGISTRY[func_name]
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


# Simulates a GPU executing tasks one at a time
# Refer to ModuleContainer for subclassing threading.Thread
class GPUWorker(threading.Thread):
    def __init__(self, priority_queue: queue.PriorityQueue, completed_queue: queue.Queue):
        super().__init__()
        self.priority_queue = priority_queue
        self.completed_queue = completed_queue
    
    def run(self):
        while True:
            task = self.priority_queue.get()
            if task is None:  # Exit signal
                break
            thread_id = threading.get_ident()
            print(f"Worker thread {thread_id} executing task {task.args}")
            task.execute()
            self.completed_queue.put(task)


# A scheduling policy that determines the priority of a task
class SchedulingPolicy:
    def __init__(self):
        pass

    def calculate_priority(self, task: GPUTask) -> float:
        return 0


# A thread that prioritizes tasks based on the scheduling policy
class RequestPriortizer(threading.Thread):
    def __init__(self, task_pool: queue.Queue, priority_queue: queue.PriorityQueue, sched_policy: SchedulingPolicy):
        super().__init__()
        self.task_pool = task_pool
        self.priority_queue = priority_queue
        self.sched_policy = sched_policy

    def run(self):
        while True:
            task = self.task_pool.get()
            if task is None:  # Propagate exit signal
                self.priority_queue.put(None)
                break
            priority = self.sched_policy.calculate_priority(task)
            self.priority_queue.put((priority, task))


# NOTE: currently we do not consider batching and the batch size is always 1
# A thread that handles incoming connections from clients
# deserialized tasks and adds them to the task queue
class ConnectionHandler(threading.Thread):
    def __init__(self, port: int, task_pool: queue.Queue, model: MultiTaskModel, prof_results: ProfilingResults):
        super().__init__()
        self.task_pool = task_pool
        self.port = port
        self.model = model
        self.prof_results = prof_results

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('localhost', self.port))
            sock.listen()
            while True:
                conn, addr = sock.accept()
                request_json = conn.recv(1024).decode().strip()
                if not request_json:  # Exit signal
                    break
                request = InferRequest.from_json(json.loads(request_json))
                stage = self.model.get_stage(request.task_name, request.next_stage_idx)
                task = GPUTask(request, "simulated_execution", args=(stage, 1, self.prof_results))
                self.task_pool.put(task)
                
                # Send a response to the client or the upstream server
                # this does not indicate completion of the task but rather that the task has been received
                conn.sendall(b"OK")
                conn.close()


# A routing policy that determines which downstream server to send a request to
# based on information from the DHT
class RoutingPolicy:
    def __init__(self, dht: DistributedHashTable, update_interval: int):
        self.dht = dht
        self.update_interval = update_interval
        self.last_update = 0
        self.update_if_needed()

    def route(self, request: InferRequest) -> int:
        # Get all servers currently serving needed stage
        possible_servers = dht.get_servers_with_stage(request.stage_index);
        # Return the server with the smallest load
        possible_servers.sort(key = lambda x: x.load_level())
        return possible_servers[0]
    
    def _update(self):
        pass

    def update_if_needed(self):
        current_time = time.time()
        if current_time - self.last_update > self.update_interval:
            self._update()
        self.last_update = time.time()


class RequestRouter(threading.Thread):
    def __init__(self, completed_queue: queue.Queue, routing_policy: RoutingPolicy, latency_est: LatencyEstimator, 
                 my_location: Point, dht: DistributedHashTable, model: MultiTaskModel):
        super().__init__()
        self.completed_queue = completed_queue
        self.routing_policy = routing_policy
        self.latency_est = latency_est
        self.my_location = my_location
        self.dht = dht
        self.model = model

        self.connections = {}   # connection with downstream servers

    def _connect(self, server_id: int, port: int):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', port))
        self.connections[server_id] = sock

    # TODO: disconnect from servers that are down or ...
    def _disconnect(self, server_id: int):
        self.connections[server_id].close()
        del self.connections[server_id]

    def _simulate_comm_latency(self, location: Point):
        comm_latency = self.latency_est.predict(self.my_location, location)
        time.sleep(comm_latency / 1000)

    def _respond_to_client(self, request: InferRequest, result: Any):
        # Simulate communication latency
        self._simulate_comm_latency(request.client_location)

        # Send the response to the client
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((request.client_ip, request.client_port))
            response = InferResponse(request, result)
            sock.sendall(json.dumps(response.to_json()).encode())

    def run(self):
        while True:
            # Periodically update the routing policy
            self.routing_policy.update_if_needed()
            
            # Check if there are any completed tasks
            # We avoid blocking the thread because we want to periodically update the routing policy
            try:
                task: GPUTask = self.completed_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Update the request
            request = task.request.update()  # that we have executed the current stage

            # If the task is the last stage, send the response to the client
            if task.request.next_stage_idx == self.model.get_task_num_stages(task.request.task_name):
                self._respond_to_client(task.request, task.result)
                continue

            # Determine the downstream server and send the request
            server_id = self.routing_policy.route(task.request)
            if server_id not in self.connections:
                self._connect(server_id, Server.START_PORT + server_id)
            sock = self.connections[server_id]

            # Simulate communication latency
            server_location = self.dht.get_server_location(server_id)
            self._simulate_comm_latency(server_location)

            # Forward the request to the downstream server
            sock.sendall(json.dumps(request.to_json()).encode())
        

class StageAssignmentPolicy:
    def __init__(self, dht: DistributedHashTable):  # TODO: server info
        self.dht = dht

    def assign_stages(self, current_stages: list[int]) -> list[int]:
        raise NotImplementedError
    

class DHTAnnouncer(threading.Thread):
    def __init__(self, server: "Server", dht: DistributedHashTable, announce_interval: float):
        super().__init__()
        self.dht = dht
        self.server = server
        self.announce_interval = announce_interval

    def _announce(self):
        pass

    def run(self):
        while True:
            self._announce()
            time.sleep(self.announce_interval)


class StageRebalancer(threading.Thread):
    def __init__(self, server: "Server", dht: DistributedHashTable, stage_assignment_policy: StageAssignmentPolicy, rebalance_interval: float):
        super().__init__()
        self.dht = dht
        self.server = server
        self.stage_assignment_policy = stage_assignment_policy
        self.rebalance_interval = rebalance_interval

    def run(self):
        while True:
            stage_ids = []  # TODO: calculate using self.server.hosted_stages
            new_stages = self.stage_assignment_policy.assign_stages(stage_ids)
            # TODO: install new stages and remove old stages
            time.sleep(self.rebalance_interval)


class ServerStatus(Enum):
    OFFLINE = 0
    ONLINE = 1


# A server that potentially hosts multiple stages of the multi-task model
# We assume that the server has a single GPU that can only execute one stage at a time
# Each server is assigned a virtual location to simulate the communication latency between servers
class Server:
    START_PORT = 8000
    
    def __init__(self, 
                 server_id: int, 
                 location: Point,
                 dht: DistributedHashTable,
                 model: MultiTaskModel,
                 prof_results: ProfilingResults,
                 latency_est: LatencyEstimator,
                 num_router_threads: int,
                 announce_interval: float,
                 rebalance_interval: float,
                 sched_policy: SchedulingPolicy, 
                 routing_policy: RoutingPolicy, 
                 stage_assignment_policy: StageAssignmentPolicy):
        
        self.server_id = server_id
        self.location = location
        self.dht = dht
        self.model = model
        self.prof_results = prof_results
        self.latency_est = latency_est
        self.announce_interval = announce_interval
        self.rebalance_interval = rebalance_interval
        self.num_router_threads = num_router_threads
        
        self.status = ServerStatus.OFFLINE
        self.hosted_stages: list[int] = list()
        
        # policies controlling the behavior of the server
        self.sched_policy = sched_policy
        self.routing_policy = routing_policy
        self.stage_assignment_policy = stage_assignment_policy
        
        # queues for communication between threads
        self.task_pool = queue.Queue()
        self.priority_queue = queue.PriorityQueue()
        self.completed_tasks = queue.Queue()

        # requests -> connection handler -> task pool
        self.connection_handler = ConnectionHandler(
            port=Server.START_PORT + server_id, 
            task_pool=self.task_pool,
            model=self.model,
            prof_results=self.prof_results,
        )

        # task pool -> request priortizer -> priority queue
        self.request_priortizer = RequestPriortizer(
            task_pool=self.task_pool, 
            priority_queue=self.priority_queue, 
            sched_policy=self.sched_policy
        )

        # priority queue -> gpu worker -> completed queue
        self.gpu_worker = GPUWorker(
            priority_queue=self.priority_queue, 
            completed_queue=self.completed_tasks
        )

        # completed queue -> request router -> downstream servers
        self.request_routers = [
            RequestRouter(
                completed_queue=self.completed_tasks,
                routing_policy=self.routing_policy,
                latency_est=self.latency_est,
                my_location=self.location,
                dht=self.dht,
                model=self.model,
            )
            for _ in range(self.num_router_threads)
        ]

        # in the background, periodically announce the server's information to the DHT
        # e.g. the stages hosted by the server, the server's status and load level, etc.
        self.dht_announcer = DHTAnnouncer(
            server=self,
            dht=self.dht, 
            announce_interval=self.announce_interval
        )

        # in the background, periodically rebalance the stages across the servers
        # e.g. if a server is overloaded, it may be assigned fewer stages
        # e.g. if some servers went offline, the remaining servers may be assigned more stages
        self.stage_rebalancer = StageRebalancer(
            server=self,
            dht=self.dht,
            stage_assignment_policy=self.stage_assignment_policy,
            rebalance_interval=self.rebalance_interval
        )

    @property
    def load_level(self):
        return self.task_pool.qsize()

    # Called when the server joins the swarm
    def join(self):
        init_stages = self.stage_assignment_policy.assign_stages(current_stages=[])
        self.hosted_stages = init_stages
        self.status = ServerStatus.ONLINE

    # Called when the server leaves the swarm
    def leave(self):
        # TODO: wait until all tasks are completed
        self.status = ServerStatus.OFFLINE
        self.hosted_stages.clear()
        # TODO: announce the server's status to the DHT

    def start(self):
        self.connection_handler.start()
        self.request_priortizer.start()
        self.gpu_worker.start()
        for router in self.request_routers:
            router.start()
        self.dht_announcer.start()
        self.stage_rebalancer.start()

    def stop(self):
        # TODO: send a signal to all threads to stop
        self.connection_handler.join()
        self.request_priortizer.join()
        self.gpu_worker.join()
        for router in self.request_routers:
            router.join()
        self.dht_announcer.join()
        self.stage_rebalancer.join()
