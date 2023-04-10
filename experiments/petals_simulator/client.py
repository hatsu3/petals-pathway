import socket
import asyncio
import json
import uuid
import random
from enum import Enum
from typing import Tuple

from geopy import Point

from server import Server
from dht import DistributedHashTable
from multitask_model import MultiTaskModel
from messages import InferRequest, InferResponse
from latency_estimator import LatencyEstimator


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


# NOTE: for now we assume that a client only requests one task repeatedly
class AsyncClient:
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
        
        self.pending_requests = set()
        self.is_running = True
    
    def get_request_interval(self):
        if self.request_mode == RequestMode.UNIFORM:
            return self.request_avg_interval
        elif self.request_mode == RequestMode.POISSON:
            # request_interval means the average interval between requests
            return random.expovariate(1 / self.request_avg_interval)
        else:
            raise ValueError(f"Invalid request mode: {self.request_mode}")

    """
    Send an already created request to the server designated by `server_ip` and
    `server_port`.
    """
    async def send_request(self, server_id: int, request: InferRequest):
        # Simulate communication latency
        server_ip, server_port = self.dht.get_server_ip_port(server_id)
        server_location = self.dht.get_server_location(server_id)
        comm_latency = self.latency_est.predict(self.location, server_location)
        await asyncio.sleep(comm_latency)

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

    async def connection_handler(self, reader, writer):
        while self.is_running:
            data = await reader.read(1024)
            if not data:
                break

            # Get the server id from the port number of the server
            remote_addr = writer.get_extra_info('peername')
            assert remote_addr is not None
            server_ip, server_port = remote_addr
            server_id = self.dht.get_server_id_by_ip_port(server_ip, server_port)

            # Parse the response and notify the client
            response = InferResponse.from_json(json.loads(data.decode("utf-8")))
            self.pending_requests.remove(response.request_id)
            print(f"Received response {response.request_id} from server {server_id}")

    async def start_server(self):
        server = await asyncio.start_server(self.connection_handler, host='0.0.0.0', port=self.recv_port)
        async with server:
            await server.serve_forever()

    async def periodic_request(self):
        while self.is_running:

            # Create a new request ID and add it to pending requests set.
            request_id = uuid.uuid4()
            self.pending_requests.add(request_id)

            # Use the ID to build a `Request` object.
            request = InferRequest(request_id, self.ip, self.recv_port, self.location, self.task_name)

            # Select the server that will receive new request.
            server_id = self.server_sel_policy.choose_server(request)

            asyncio.create_task(self.send_request(server_id, request))

            await asyncio.sleep(self.get_request_interval())

    async def update_server_sel_policy(self):
        while self.is_running:
            self.server_sel_policy.update()
            await asyncio.sleep(self.update_interval)

    async def run(self):
        await asyncio.gather(
            self.periodic_request(),
            self.update_server_sel_policy(),
            self.start_server(),
            return_exceptions=True
        )

    def stop(self):
        self.is_running = False
