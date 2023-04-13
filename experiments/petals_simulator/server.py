import json
import queue
import socket
import threading
import time
import logging
from typing import Any
from abc import abstractmethod

from geopy import Point

import random

from multitask_model import MultiTaskModel, Stage
from latency_estimator import LatencyEstimator
from dht import DistributedHashTable, ServerStatus, ServerNonExistentException
from messages import InferRequest, InferResponse
from stage_profiler import ProfilingResults

from trace_visualizer import TraceVisualizer


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
    @TraceVisualizer(log_file_path='trace.json')
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
    def __init__(self, server: "Server"):
        super().__init__()
        self.server = server
        self.priority_queue = server.priority_queue
        self.completed_queue = server.completed_tasks
    
    def run(self):
        while self.server.is_running:
            try:
                priority, timestamp, task = self.priority_queue.get(timeout=5)
            except queue.Empty:
                continue
            if task is None:  # Exit signal
                break
            thread_id = threading.get_ident()
            logging.info(f"Worker thread {thread_id} executing task {task.args}")
            task.execute()
            self.completed_queue.put(task)


# A scheduling policy that determines the priority of a task
class SchedulingPolicy:
    def __init__(self, model: MultiTaskModel):
        self.model = model

    @abstractmethod
    def calculate_priority(self, task: GPUTask) -> float:
        pass

class BaselineSchedulingPolicy:
    def __init__(self, model: MultiTaskModel):
        self.model = model

    @abstractmethod
    def calculate_priority(self, task: GPUTask) -> float:
        return random.uniform(0, 100)

class SchedulingEstimationPolicy(SchedulingPolicy):
    def __init__(self, model: MultiTaskModel, profiling_results: ProfilingResults):
        super().__init__(model)
        self.profiling_results = profiling_results
    
    def estimate_time_to_completion(self, task: GPUTask):
        estimation = 0.0
        task_name = task.request.task_name
        stage_name = self.model.get_stage(task_name, task.request.next_stage_idx).name
        while stage_name is not None:
            estimation += self.profiling_results.get_latency(stage_name, batch_size=1)
            stage_name = self.model.get_next_stage(stage_name, task_name)
            
        return estimation
    
    def calculate_priority(self, task: GPUTask) -> float:
        # estimated_completion_time = current_time - timestamp + estimate_time_to_completion
        # the lower priority, the earlier to be scheduled, so negate this expression
        return -(time.time() * 1e3 - task.request.timestamp + self.estimate_time_to_completion(task))


# A thread that prioritizes tasks based on the scheduling policy
class RequestPriortizer(threading.Thread):
    def __init__(self, server: "Server"):
        super().__init__()
        self.server = server
        self.task_pool = server.task_pool
        self.priority_queue = server.priority_queue
        self.sched_policy = server.sched_policy

    def run(self):
        while self.server.is_running:
            try:
                task = self.task_pool.get(timeout=5)
            except queue.Empty:
                continue
            if task is None:  # Propagate exit signal
                self.priority_queue.put(None)
                break
            priority = self.sched_policy.calculate_priority(task)
            # add a timestamp to achieve tie-breaking
            self.priority_queue.put((priority, time.time(), task))


# NOTE: currently we do not consider batching and the batch size is always 1
# A thread that handles incoming connections from clients
# deserialized tasks and adds them to the task queue
class ConnectionHandler(threading.Thread):
    def __init__(self, server: "Server"):
        super().__init__()
        self.server = server
        self.task_pool = server.task_pool
        self.port = server.port
        self.model = server.model
        self.prof_results = server.prof_results

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('localhost', self.port))
            sock.listen()
            sock.settimeout(5.0)
            while self.server.is_running:
                try:
                    conn, addr = sock.accept()
                    request_json = conn.recv(1024).decode().strip()
                    logging.debug(f"Server {self.server.server_id} receiving request {request_json}.")
                    if not request_json:  # Exit signal
                        break
                    request = InferRequest.from_json(json.loads(request_json))
                    logging.info(f"Server {self.server.server_id} receives request {request.request_id}.")
                    stage = self.model.get_stage(request.task_name, request.next_stage_idx)
                    task = GPUTask(request, "simulated_execution", args=(stage, 1, self.prof_results))
                    self.task_pool.put(task)
                    
                    # Send a response to the client or the upstream server
                    # this does not indicate completion of the task but rather that the task has been received
                    conn.sendall(b"OK")
                    conn.close()
                except socket.timeout:
                    continue


# A routing policy that determines which downstream server to send a request to
# based on information from the DHT
class RoutingPolicy:
    def __init__(self, model: MultiTaskModel, dht: DistributedHashTable, update_interval: int):
        self.model = model
        self.dht = dht
        self.update_interval = update_interval
        self.last_update = 0
        self.update_if_needed()

    def route(self, request: InferRequest) -> int:
        # Get all servers currently serving needed stage
        next_stage = self.model.get_stage(request.task_name, request.next_stage_idx)
        possible_servers = self.dht.get_servers_with_stage(next_stage.name)

        if len(possible_servers) > 0:
            # Return the server with the smallest load
            possible_servers.sort(key = lambda x: self.dht.get_server_load(x))
            return possible_servers[0]
        else:
            return -1
    
    def _update(self):
        pass

    def update_if_needed(self):
        current_time = time.time()
        if current_time - self.last_update > self.update_interval:
            self._update()
        self.last_update = time.time()

# Baseline (random) routing policy
class RandomRoutingPolicy:
    # Update interval is probably not used, but keeping things uniform
    def __init__(self, model: MultiTaskModel, dht: DistributedHashTable, update_interval: int):
        self.model = model
        self.dht = dht
        self.update_interval = update_interval
        self.last_update = 0
        self.update_if_needed()

    def route(self, request: InferRequest) -> int:
        # Get all servers serving a given stage
        next_stage = self.model.get_stage(request.task_name, request.next_stage_idx)
        possible_servers = self.dht.get_servers_with_stage(next_stage.name)

        if len(possible_servers) > 0:
            # Return random server in the list
            return random.choice(possible_servers)

    def _update(self):
        pass

    def update_if_needed(self):
        current_time = time.time()
        if current_time - self.last_update > self.update_interval:
            self._update()
        self.last_update = time.time()

class RequestRouter(threading.Thread):
    def __init__(self, server: "Server"):
        super().__init__()
        self.server = server
        self.completed_queue = server.completed_tasks
        self.routing_policy = server.routing_policy
        self.latency_est = server.latency_est
        self.my_location = server.location
        self.dht = server.dht
        self.model = server.model

    def _connect(self, server_id: int):
        server_ip, server_port = self.dht.get_server_ip_port(server_id)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server_ip, server_port))
        return sock

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
        logging.info(f"Server {self.server.server_id} responded to {request.request_id}.")

    def run(self):
        while self.server.is_running:
            # Periodically update the routing policy
            self.routing_policy.update_if_needed()
            
            # Check if there are any completed tasks
            # We avoid blocking the thread because we want to periodically update the routing policy
            try:
                task: GPUTask = self.completed_queue.get(timeout=1)
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
            
            if server_id < 0:
                time.sleep(1)
                continue

            if server_id != self.server.server_id:
                sock = self._connect(server_id)

                # Simulate communication latency
                server_location = self.dht.get_server_location(server_id)
                self._simulate_comm_latency(server_location)

                # Forward the request to the downstream server
                try:
                    sock.sendall(json.dumps(request.to_json()).encode())
                except ConnectionResetError:
                    # TODO: fault tolerance
                    continue
            else:
                stage = self.model.get_stage(request.task_name, request.next_stage_idx)
                task = GPUTask(request, "simulated_execution", args=(stage, 1, self.server.prof_results))
                self.server.task_pool.put(task)
        

class StageAssignmentPolicy:
    def __init__(self, model: MultiTaskModel, dht: DistributedHashTable):
        self.model = model
        self.dht = dht

    @abstractmethod
    def assign_stages(self, current_stages: list[str]) -> list[str]:
        pass


class BaselineStageAssignmentPolicy(StageAssignmentPolicy):
    def assign_stages(self, current_stages: list[str]) -> list[str]:
        stages = self.model.get_stages()
        capabilities = {stage.name: len(self.dht.get_servers_with_stage(stage.name)) for stage in stages}
        average_load = len(stages) / self.dht.get_number_of_servers()
        while average_load > len(current_stages):
            # pick the stage with the least number of servers
            candidate = min(capabilities, key=capabilities.get) # type: ignore
            current_stages.append(candidate)
            del capabilities[candidate]
        return current_stages


class DHTAnnouncer(threading.Thread):
    def __init__(self, server: "Server"):
        super().__init__()
        self.server = server
        self.dht = server.dht
        self.announce_interval = server.announce_interval

    """
    Let the DHT know of the current status of this server.
    """
    def _announce(self):
        # List all the information that potentially needs to be sent to DHT
        # NOTE: For now, we simply refresh all the information every time (at least it does not hurt)
        server_id = self.server.server_id
        assert server_id is not None
        self.dht.modify_server_info(server_id, "ip", self.server.ip)
        self.dht.modify_server_info(server_id, "port", self.server.port)
        self.dht.modify_server_info(server_id, "location", self.server.location)
        self.dht.modify_server_info(server_id, "stages", self.server.hosted_stages)
        self.dht.modify_server_info(server_id, "load", self.server.load_level)
        self.dht.modify_server_info(server_id, "status", ServerStatus.ONLINE)

    def run(self):
        while self.server.is_running:
            self._announce()
            time.sleep(self.announce_interval)


class StageRebalancer(threading.Thread):
    def __init__(self, server: "Server"):
        super().__init__()
        self.server = server
        self.dht = server.dht
        self.stage_assignment_policy = server.stage_assignment_policy
        self.rebalance_interval = server.rebalance_interval

    def run(self):
        while self.server.is_running:
            stage_ids = []
            new_stages = self.stage_assignment_policy.assign_stages(stage_ids)
            self.hosted_stages = new_stages
            assert self.server.server_id is not None
            try:
                self.dht.modify_server_info(self.server.server_id, "stages", self.hosted_stages)
            except ServerNonExistentException:
                continue
            time.sleep(self.rebalance_interval)


# A server that potentially hosts multiple stages of the multi-task model
# We assume that the server has a single GPU that can only execute one stage at a time
# Each server is assigned a virtual location to simulate the communication latency between servers
class Server:
    def __init__(self, 
                 ip: str,
                 port: int,
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
        
        logging.info(f"A new Server is being initiated.")

        # server's configurations
        self.ip = ip
        self.port = port
        self.location = location
        self.dht = dht
        self.model = model
        self.prof_results = prof_results
        self.latency_est = latency_est
        self.announce_interval = announce_interval
        self.rebalance_interval = rebalance_interval
        self.num_router_threads = num_router_threads
        
        # policies controlling the behavior of the server
        self.sched_policy = sched_policy
        self.routing_policy = routing_policy
        self.stage_assignment_policy = stage_assignment_policy

        # the server's id and the stages it hosts
        self.server_id = self.dht.register_server()
        self.hosted_stages: list[str] = list()
        
        # queues for communication between threads
        self.task_pool = queue.Queue()
        self.priority_queue = queue.PriorityQueue()
        self.completed_tasks = queue.Queue()

        # shared flag for signaling child threads
        self.is_running = False

        """ child threads """
        
        # requests -> connection handler -> task pool
        self.connection_handler = ConnectionHandler(self)

        # task pool -> request priortizer -> priority queue
        self.request_priortizer = RequestPriortizer(self)

        # priority queue -> gpu worker -> completed queue
        self.gpu_worker = GPUWorker(self)

        # completed queue -> request router -> downstream servers
        self.request_routers = [
            RequestRouter(self)
            for _ in range(self.num_router_threads)
        ]

        # in the background, periodically announce the server's information to the DHT
        # e.g. the stages hosted by the server, the server's status and load level, etc.
        self.dht_announcer = DHTAnnouncer(self)

        # in the background, periodically rebalance the stages across the servers
        # e.g. if a server is overloaded, it may be assigned fewer stages
        # e.g. if some servers went offline, the remaining servers may be assigned more stages
        self.stage_rebalancer = StageRebalancer(self)

        logging.info(f"Server {self.server_id} initiated.")

    @property
    def load_level(self):
        return self.task_pool.qsize()

    def start(self):
        logging.info(f"Server {self.server_id} is starting.")
        
        # set the shared flag to start all threads
        self.is_running = True

        # start all threads
        self.connection_handler.start()
        self.request_priortizer.start()
        self.gpu_worker.start()
        for router in self.request_routers:
            router.start()
        self.dht_announcer.start()
        self.stage_rebalancer.start()

        init_stages = self.stage_assignment_policy.assign_stages(current_stages=[])
        self.hosted_stages = init_stages

        logging.info(f"Server {self.server_id} started.")

    def stop(self):
        logging.info(f"Server {self.server_id} is stopping.")

        self.dht.delete_server(self.server_id)

        # set the shared flag to stop all threads
        self.is_running = False

        # wait for all threads to finish
        self.connection_handler.join()
        logging.debug(f"Server {self.server_id} stopped the connection handler.")
        self.request_priortizer.join()
        logging.debug(f"Server {self.server_id} stopped the request prioritizer.")
        self.gpu_worker.join()
        logging.debug(f"Server {self.server_id} stopped the gpu worker.")
        for router in self.request_routers:
            router.join()
        logging.debug(f"Server {self.server_id} stopped the requster routers.")
        self.dht_announcer.join()
        logging.debug(f"Server {self.server_id} stopped the dht announcer.")
        self.stage_rebalancer.join()
        logging.debug(f"Server {self.server_id} stopped the stage rebalancer.")

        assert self.server_id is not None
        self.hosted_stages.clear()

        logging.info(f"Server {self.server_id} stopped.")

    def run(self, run_time: float):
        self.start()

        if run_time > 0:
            time.sleep(run_time)
            self.stop()
