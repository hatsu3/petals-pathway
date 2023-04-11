import queue
import socket
import json
import uuid
import random
from enum import Enum
import threading
import time

from geopy import Point

from dht import DistributedHashTable
from multitask_model import MultiTaskModel
from messages import InferRequest, InferResponse
from latency_estimator import LatencyEstimator

import logging

class RequestMode(Enum):
    UNIFORM = 1
    POISSON = 2


# Policy for selecting which server in the swarm to send a request to
# options: random, closest, least loaded, etc.
class ServerSelectionPolicy:
    def __init__(self, model: MultiTaskModel, dht: DistributedHashTable):
        self.model = model
        self.dht = dht

    """
    Choose the server that can serve the first stage of the new request. 
    Return the index of the best server
    """
    def choose_server(self, request: InferRequest) -> int:
        # Pick the next stage of the model that is currently running.
        next_stage = self.model.get_stage(request.task_name, request.next_stage_idx)
        # Find all the servers serving that stage.
        possible_servers = self.dht.get_servers_with_stage(next_stage.name)
        # Find the server with smallest current load, and return its IP and port.
        possible_servers.sort(key = lambda x: self.dht.get_server_load(x))
        return possible_servers[0]

    def update(self):
        pass


class Client:
    def __init__(self, 
                 ip: str,
                 send_port: int,
                 recv_port: int,
                 client_id: int, 
                 location: Point, 
                 task_name: str,
                 dht: DistributedHashTable, 
                 model: MultiTaskModel,
                 latency_est: LatencyEstimator,
                 server_sel_policy: ServerSelectionPolicy, 
                 request_mode=RequestMode.POISSON,
                 request_avg_interval=5, 
                 update_interval=10):
        
        self.ip = ip
        self.send_port = send_port
        self.recv_port = recv_port
        self.client_id = client_id
        self.location = location
        self.task_name = task_name
        self.dht = dht
        self.model = model
        self.latency_est = latency_est
        self.server_sel_policy = server_sel_policy
        self.request_mode = request_mode
        self.request_avg_interval = request_avg_interval
        self.update_interval = update_interval
        
        self.is_running = True
        self.pending_requests = set()
        self.response_queue = queue.Queue()

    def get_request_interval(self):
        if self.request_mode == RequestMode.UNIFORM:
            return self.request_avg_interval
        elif self.request_mode == RequestMode.POISSON:
            # request_interval means the average interval between requests
            return random.expovariate(1 / self.request_avg_interval)
        else:
            raise ValueError(f"Invalid request mode: {self.request_mode}")
    
    def send_request(self, server_id: int, request: InferRequest):
        # Simulate communication latency
        server_ip, server_port = self.dht.get_server_ip_port(server_id)
        server_location = self.dht.get_server_location(server_id)
        comm_latency = self.latency_est.predict(self.location, server_location)
        time.sleep(comm_latency)

        # Get the actual bytes from the request.
        request_bytes = json.dumps(request.to_json()).encode("utf-8")
        
        # Send request to the entry server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((server_ip, server_port))
            sock.sendall(request_bytes)
            sock.settimeout(1)
            try:
                response = sock.recv(1024).decode("utf-8")
                if response != "OK": 
                    print(f"Received error response from entry server {server_id}")
            except socket.timeout:
                print(f"Request {request.request_id} timed out")

        print(f"Request {request.request_id} sent to server {server_id}")

    def receive_responses(self):
        while self.is_running:
            try:
                conn, addr = self.response_queue.get(timeout=1)
                data = conn.recv(1024)
                if not data:
                    break

                # Get the server id from the port number of the server
                server_ip, server_port = addr
                server_id = self.dht.get_server_id_by_ip_port(server_ip, server_port)

                # Parse the response and notify the client
                response = InferResponse.from_json(json.loads(data.decode("utf-8")))
                self.pending_requests.remove(response.request_id)
                print(f"Received response {response.request_id} from server {server_id}")

            except queue.Empty:
                continue

    def connection_handler(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('0.0.0.0', self.recv_port))
        server_socket.listen(1)

        while self.is_running:
            conn, addr = server_socket.accept()
            self.response_queue.put((conn, addr))

        server_socket.close()

    def send_requests(self):
        while self.is_running:

            # Create a new request ID and add it to pending requests set.
            request_id = uuid.uuid4()
            self.pending_requests.add(request_id)

            # Use the ID to build a `Request` object.
            request = InferRequest(request_id, self.ip, self.recv_port, self.location, self.task_name)

            # Select the server that will receive new request.
            server_id = self.server_sel_policy.choose_server(request)

            self.send_request(server_id, request)

            time.sleep(self.get_request_interval())

    def update_server_sel_policy(self):
        while self.is_running:
            self.server_sel_policy.update()
            time.sleep(self.update_interval)

    def run(self, run_time: float):
        request_thread = threading.Thread(target=self.send_requests)
        listener_thread = threading.Thread(target=self.connection_handler)
        response_thread = threading.Thread(target=self.receive_responses)
        update_policy_thread = threading.Thread(target=self.update_server_sel_policy)

        request_thread.start()
        listener_thread.start()
        response_thread.start()
        update_policy_thread.start()

        if run_time > 0:
            time.sleep(run_time)
            self.stop()

        request_thread.join()
        listener_thread.join()
        response_thread.join()
        update_policy_thread.join()

    def stop(self):
        self.is_running = False
