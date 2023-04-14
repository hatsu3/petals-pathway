import queue
import socket
import json
import uuid
import random
from enum import Enum
import threading
import time
import os

from geopy import Point

from dht import DistributedHashTable, ServerNonExistentException
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
        # Find all the servers serving that stage and are online.
        possible_servers = self.dht.get_servers_with_stage(next_stage.name)

        if len(possible_servers) > 0:
            # Find the server with smallest current load, and return its IP and port.
            possible_servers.sort(key = lambda x: self.dht.get_server_load(x))
            return possible_servers[0]
        else:
            return -1

    def update(self):
        pass


class Client:
    def __init__(self, 
                 ip: str,
                 port: int,
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
        self.port = port
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
        self.pending_requests = {}
        self.response_queue = queue.Queue()

        logging.info(f"Client {self.client_id} initialized to {self.location}.")

    def get_request_interval(self):
        if self.request_mode == RequestMode.UNIFORM:
            return self.request_avg_interval
        elif self.request_mode == RequestMode.POISSON:
            # request_interval means the average interval between requests
            return random.expovariate(1 / self.request_avg_interval)
        else:
            raise ValueError(f"Invalid request mode: {self.request_mode}")
    
    def send_request(self, request: InferRequest):
        server_id = -1
        while self.is_running:
            try:
                server_id = self.server_sel_policy.choose_server(request)
                assert server_id is not None
                logging.debug(f"Client {self.client_id} is sending server {server_id} request {request.request_id} .")

                # Simulate communication latency
                server_ip, server_port = self.dht.get_server_ip_port(server_id)
                server_location = self.dht.get_server_location(server_id)
                comm_latency = self.latency_est.predict(self.location, server_location)
                logging.debug(f"Predicted communication latency: {comm_latency}.")
                time.sleep(comm_latency / 1000)

                # Get the actual bytes from the request.
                request_bytes = json.dumps(request.to_json()).encode("utf-8")
                
                # Send request to the entry server
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.connect((server_ip, server_port))
                    sock.sendall(request_bytes)
                    sock.settimeout(5)
                    try:
                        response = sock.recv(1024).decode("utf-8")
                        if response != "OK": 
                            logging.warning(f"Client {self.client_id} received error response from entry server {server_id} for request {request.request_id}.")
                    except ConnectionResetError:
                        continue
                    except socket.timeout:
                        logging.warning(f"Client {self.client_id} sending to server {server_id} request {request.request_id} timed out.")
                        continue
                logging.info(f"Client {self.client_id} sent server {server_id} request {request.request_id}.")
                break
            except ConnectionRefusedError:
                continue
            except ConnectionResetError:
                continue
            except ServerNonExistentException:
                continue

    def receive_responses(self):
        while self.is_running:
            try:
                conn, addr = self.response_queue.get(timeout=1)
                data = conn.recv(1024)
                response_timestamp = time.time()
                if not data:
                    break

                # Parse the response and notify the client
                response = InferResponse.from_json(json.loads(data.decode("utf-8")))
                server_id = response.responser_server_id
                end_to_end_latency = response_timestamp - self.pending_requests[response.request_id]
                del self.pending_requests[response.request_id]
                logging.info(f"Client {self.client_id} received response from server {server_id} for request {response.request_id}.")

                # Create the `e2e_latency` directory if it is not there
                os.makedirs(f"e2e_latency", exist_ok=True)
                # collecting end-to-end latency information for evaluation part
                with open(f"e2e_latency/{self.client_id}.csv", "a") as f:
                    f.write(f"{self.client_id}, {end_to_end_latency}\n")

            except queue.Empty:
                continue

            except socket.timeout:
                continue

    def connection_handler(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(1)
        server_socket.settimeout(5.0)

        while self.is_running:
            try:
                conn, addr = server_socket.accept()
                self.response_queue.put((conn, addr))
            except socket.timeout:
                continue

        server_socket.close()

    def send_requests(self):
        while self.is_running:

            # Create a new request ID and add it to pending requests set.
            request_id = uuid.uuid4()

            # Use the ID to build a `Request` object.
            request = InferRequest(
                request_id, self.ip, self.port, self.location, 
                forwarder_server_id=None, task_name=self.task_name
            )

            # # Select the server that will receive new request.
            # server_id = self.server_sel_policy.choose_server(request)

            # if server_id >= 0:
            self.send_request(request)

            # TODO: retry for pending_requests (timeout & server not found)
            self.pending_requests[request_id] = time.time()

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
        # logging.debug(f"Client {self.client_id} stopped the request thread.")
        listener_thread.join()
        # logging.debug(f"Client {self.client_id} stopped the listener thread.")
        response_thread.join()
        # logging.debug(f"Client {self.client_id} stopped the response thread.")
        update_policy_thread.join()
        # logging.debug(f"Client {self.client_id} stopped the updating-policy thread.")

    def stop(self):
        self.is_running = False
