from concurrent.futures import ThreadPoolExecutor
import sys
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

from trace_visualizer import TraceVisualizer, DIVISOR


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
            # logging.debug(f"Worker thread {thread_id} executing task {task.args}")
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
        return -(time.time() / 1e3 - task.request.timestamp / 1e3 + self.estimate_time_to_completion(task))


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
    def __init__(self, server: "Server", num_workers: int = 20):
        super().__init__()
        self.server = server
        self.task_pool = server.task_pool
        self.port = server.port
        self.model = server.model
        self.prof_results = server.prof_results
        self.num_workers = num_workers

    def _handle_connection(self, conn: socket.socket, addr: str):
        request_json = conn.recv(1024).decode().strip()
        logging.debug(f"Server {self.server.server_id} receiving request {request_json}.")

        request = InferRequest.from_json(json.loads(request_json))
        logging.debug(f"Server {self.server.server_id} receives request {request.request_id}.")
        
        # determine if the request is from a client or from an upstream server
        sender_server_id = request.forwarder_server_id
        if sender_server_id is None:  # from a client
            logging.debug(f"Server {self.server.server_id} receives request (task={request.task_name}, stage={request.next_stage_idx}) from a client.")
        else:  # from an upstream server
            logging.debug(f"Server {self.server.server_id} receives request (task={request.task_name}, stage={request.next_stage_idx}) from server {sender_server_id}.")
        
        # Add the task to the task pool
        stage = self.model.get_stage(request.task_name, request.next_stage_idx)
        task = GPUTask(request, "simulated_execution", args=(stage, 1, self.prof_results))
        self.task_pool.put(task)
        
        # Update the request count
        self.server.request_within_last_interval += 1
        if stage.name not in self.server.request_within_last_interval_per_stage:
            self.server.request_within_last_interval_per_stage[stage.name] = 0
        self.server.request_within_last_interval_per_stage[stage.name] += 1
        
        # Send a response to the client or the upstream server
        # this does not indicate completion of the task but rather that the task has been received
        conn.sendall(b"OK")
        conn.close()

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('localhost', self.port))
            sock.listen()
            sock.settimeout(5.0)

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                while self.server.is_running:
                    try:
                        conn, addr = sock.accept()
                        executor.submit(self._handle_connection, conn, addr)
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
            try:
                possible_servers.sort(key = lambda x: self.dht.get_server_instant_load(x))
            except ServerNonExistentException:
                return -1
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
        else:
            return -1

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
        try:
            server_ip, server_port = self.dht.get_server_ip_port(server_id)
        except ServerNonExistentException:
            raise ServerNonExistentException
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
            try:
                sock.connect((request.client_ip, request.client_port))
                response = InferResponse(request, result, self.server.server_id)
                sock.sendall(json.dumps(response.to_json()).encode())
                logging.debug(f"Server {self.server.server_id} responded to {request.request_id}.")
            except ConnectionResetError:
                pass

    def _forward_request(self, request: InferRequest, server_id: int):
        logging.debug(
            f"Server {self.server.server_id} is forwarding request (task={request.task_name}, "
            f"stage={request.next_stage_idx}) to a downsteam server {server_id}"
        )
        try:
            sock = self._connect(server_id)
        except ServerNonExistentException:
            logging.warning(
                f"Server {self.server.server_id} failed to get the IP and port of server {server_id}"
                f"because the downstream server does not exist."
            )
            return

        # Simulate communication latency
        try:
            server_location = self.dht.get_server_location(server_id)
        except ServerNonExistentException:
            logging.warning(
                f"Server {self.server.server_id} failed to get the location of server {server_id}"
                f"because the downstream server does not exist."
            )
            return

        self._simulate_comm_latency(server_location)

        # Forward the request to the downstream server
        try:
            sock.sendall(json.dumps(request.to_json()).encode())
            logging.info(
                f"Server {self.server.server_id} forwarded request (task={request.task_name}, "
                f"stage={request.next_stage_idx}) to a downsteam server {server_id}"
            )
        except ConnectionResetError:
            logging.warning(
                f"Server {self.server.server_id} failed to forward request to server {server_id}"
                f"because the connection was reset by the downstream server."
            )
            # TODO: fault tolerance
            return

    def run(self):
        with ThreadPoolExecutor(max_workers=10) as executor:
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
                if task.request.next_stage_idx >= self.model.get_task_num_stages(task.request.task_name):
                    logging.info(
                        f"Server {self.server.server_id} completed request (task={request.task_name}. "
                        f"Responding to client {request.client_ip}:{request.client_port}."
                    )
                    executor.submit(self._respond_to_client, task.request, task.result)
                    continue
                
                # If the task is not the last stage, forward the request to a downstream server or itself
                # We specify the server id of the forwarder in the request because it's awkward to determine
                # the server id of the forwarder from ip and port
                request.forwarder_server_id = self.server.server_id

                # Determine the downstream server and send the request
                server_id = self.routing_policy.route(task.request)
                if server_id < 0:
                    logging.debug(
                        f"Server {self.server.server_id} failed to find a downstream server in routing"
                    )
                    time.sleep(1)
                    continue

                if server_id != self.server.server_id:
                    executor.submit(self._forward_request, request, server_id)
                else:
                    logging.info(
                        f"Server {self.server.server_id} does not forward request (task={request.task_name}, "
                        f"stage={request.next_stage_idx}) because it hosts the next stage."
                    )
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


class DummyStageAssignmentPolicy(StageAssignmentPolicy):
    def assign_stages(self, current_stages: list[str]) -> list[str]:
        return [stage.name for stage in self.model.get_stages()]
 

class BaselineStageAssignmentPolicy(StageAssignmentPolicy):
    def assign_stages(self, current_stages: list[str]) -> list[str]:
        stages = self.model.get_stages()
        serving_servers = {stage.name: len(self.dht.get_servers_with_stage(stage.name)) for stage in stages}
        number_of_servers = self.dht.get_number_of_servers()
        if number_of_servers > 0:
            for stage_name, number_of_server in serving_servers.items():
                if number_of_server < 1:
                    current_stages.append(stage_name)
            average_load = len(stages) / self.dht.get_number_of_servers()
            if average_load > len(current_stages):
                while average_load > len(current_stages):
                    # pick the stage with the least number of servers
                    candidate = min(serving_servers, key=serving_servers.get) # type: ignore
                    current_stages.append(candidate)
                    del serving_servers[candidate]
                return current_stages
            else:
                to_restore = []
                redundant = {stage: serving_servers[stage] for stage in current_stages}
                while average_load < len(current_stages) and len(redundant) != 0:
                    candidate = max(redundant, key=redundant.get)
                    current_stages.remove(candidate)
                    del redundant[candidate]
                    # flip a coin to reduce the possibility
                    # that all servers drop this stage
                    if random.randint(0, 99) % 5:
                        to_restore.append(candidate)
                current_stages.extend(to_restore)
                return current_stages
        else:
            return []


class RequestRateStageAssignmentPolicy(StageAssignmentPolicy):
    # this function is only called when a server is online or starting up (current_stages is empty)
    # which means current_stages are reflected in the DHT
    def assign_stages(self, current_stages: list[str]) -> list[str]:
        stages = self.model.get_stages()
        req_rate = self.dht.get_normalized_stage_req_rate()  # stage name -> normalized req rate

        # calculate the fulfillment score for each stage as num_replicas / req_rate
        # the lower the score, the higher priority we should allocate more servers to the stage
        def calc_fulfillment_score(stage: Stage) -> float:
            num_replicas = len(self.dht.get_servers_with_stage(stage.name))
            if req_rate[stage.name] == 0:
                return sys.maxsize  # handle zero division
            else:
                return num_replicas / req_rate[stage.name]
        
        fulfillment_scores = {
            stage.name: calc_fulfillment_score(stage) for stage in stages
        }

        number_of_servers = self.dht.get_number_of_servers()
        assert number_of_servers > 0, "There should be at least one server in the system"
        
        # we calculate the current load assuming that for each stage the load is distributed evenly
        # across all replicas
        num_stage_replicas = {
            stage.name: len(self.dht.get_servers_with_stage(stage.name)) 
            for stage in stages
        }

        current_load = sum([
            req_rate[stage] / (num_stage_replicas[stage] + 1)
            for stage in current_stages
        ])

        # serve all stages that are not currently being served by any server
        for stage_name, num_replicas in num_stage_replicas.items():
            if num_replicas == 0:
                current_stages.append(stage_name)
                current_load += req_rate[stage_name] / (num_stage_replicas[stage_name] + 1)
                del fulfillment_scores[stage_name]  # no replacement

        # our goal is to make the load of each server as close to the average load as possible
        target_load = 1 / number_of_servers
        
        # if the current load is lower than the average load, we should add more servers
        # we pick a few stages with the lowest fulfillment scores
        if target_load > current_load:
            while target_load > current_load and len(fulfillment_scores) > 0:
                candidate = min(fulfillment_scores, key=fulfillment_scores.get) # type: ignore
                current_stages.append(candidate)
                current_load += req_rate[candidate] / (num_stage_replicas[candidate] + 1)
                del fulfillment_scores[candidate]  # no replacement
        
        # if the current load is higher than the average load, we should remove some servers
        # we remove a few stages with the highest fulfillment scores
        elif target_load < current_load:
            curr_stg_scores = {
                stage: fulfillment_scores[stage] 
                for stage in current_stages
                if stage in fulfillment_scores
            }
            
            while target_load < current_load and len(curr_stg_scores) > 0:                
                candidate = max(curr_stg_scores, key=curr_stg_scores.get) # type: ignore
                # only remove the stage if there is at least one replica left
                if num_stage_replicas[candidate] >= 2:
                    current_stages.remove(candidate)
                    current_load -= req_rate[candidate] / num_stage_replicas[candidate]
                
                del curr_stg_scores[candidate]  # no replacement
        
        return current_stages


class DHTAnnouncer(threading.Thread):
    def __init__(self, server: "Server"):
        super().__init__()
        self.server = server
        self.dht = server.dht
        self.announce_interval = server.announce_interval
        self.load_window = []
        self.load_window_size = 3

    """
    Let the DHT know of the current status of this server.
    """
    def _announce(self):

        # Get the ID of the server. Make sure the number makes sense
        server_id = self.server.server_id
        assert server_id is not None
        assert server_id >= 0
        
        # If this server is shutting down, don't do anything
        if self.dht.get_server_status(server_id) == ServerStatus.STOPPING:
            return

        # List all the information that potentially needs to be sent to DHT
        # NOTE: For now, we simply refresh all the information every time (at least it does not hurt)
        self.dht.modify_server_info(server_id, "ip", self.server.ip)
        self.dht.modify_server_info(server_id, "port", self.server.port)
        self.dht.modify_server_info(server_id, "location", self.server.location)
        self.dht.modify_server_info(server_id, "stages", self.server.hosted_stages)
        self.dht.modify_server_info(server_id, "instant_load", self.server.task_pool.qsize() + self.server.priority_queue.qsize())
        self.dht.modify_server_info(server_id, "load", sum(self.load_window))
        self.dht.modify_server_info(server_id, "status", ServerStatus.ONLINE)
        self.dht.update_stage_req_rate(self.server.request_within_last_interval_per_stage)

    def run(self):
        while self.server.is_running:
            try:
                # Update the load window
                self.load_window.append(self.server.request_within_last_interval)
                if len(self.load_window) > self.load_window_size:
                    self.load_window.pop(0)
                
                # Announce the current status
                assert self.server.gpu_worker.ident is not None
                logging.debug(f"Thread {self.server.gpu_worker.ident % DIVISOR} load: {sum(self.load_window)} ({len(self.server.hosted_stages)}).")
                self._announce()
                logging.debug(f"Normalized stage request rate: {self.server.dht.get_normalized_stage_req_rate()}")

                # Reset the request counter
                self.server.request_within_last_interval = 0
                self.server.request_within_last_interval_per_stage = dict()
            except ServerNonExistentException:
                continue

            time.sleep(self.announce_interval)


class StageRebalancer(threading.Thread):
    def __init__(self, server: "Server"):
        super().__init__()
        self.server = server
        self.dht = server.dht
        self.stage_assignment_policy = server.stage_assignment_policy
        self.rebalance_interval = server.rebalance_interval

    def run(self):
        while self.server.is_running and self.dht.get_server_status(self.server.server_id) == ServerStatus.ONLINE:
            stage_ids = self.server.hosted_stages
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
        
        # logging.debug(f"A new Server is being initiated.")

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
        self.hosted_stages: list[str] = []
        
        # queues for communication between threads
        self.task_pool = queue.Queue()
        self.priority_queue = queue.PriorityQueue()
        self.completed_tasks = queue.Queue()

        self.request_within_last_interval = 0
        self.request_within_last_interval_per_stage = dict()

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

    @property
    def load_level(self):
        return self.task_pool.qsize()

    def start(self):
        # logging.debug(f"Server {self.server_id} is starting.")
        
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

        self.hosted_stages = self.stage_assignment_policy.assign_stages(list())

        logging.info(f"Server {self.server_id} started and hosting stages: {self.hosted_stages}.")

    def stop(self):
        # logging.debug(f"Server {self.server_id} is stopping.")

        # set the server's status to STOPPING so that other servers and clients will not send requests to it
        # also the announcer will stop announcing the server's information
        self.dht.modify_server_info(self.server_id, "status", ServerStatus.STOPPING)

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

        # finalize termination and remove the server from the DHT
        assert self.server_id is not None
        self.dht.delete_server(self.server_id)
        logging.debug(f"Server {self.server_id} deleted its info from dht.")
        self.hosted_stages.clear()

        logging.info(f"Server {self.server_id} stopped.")

    def run(self, run_time: float):
        self.start()

        assert self.gpu_worker.ident is not None
        logging.info(f"Server {self.server_id} (id: {self.gpu_worker.ident % DIVISOR}) started with location {self.location}.")

        if run_time > 0:
            time.sleep(run_time)
            self.stop()
