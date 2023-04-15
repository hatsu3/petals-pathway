from abc import ABC, abstractmethod
import random
import time

from dht import DistributedHashTable, ServerNonExistentException
from messages import InferRequest
from multitask_model import MultiTaskModel


class RoutingPolicy(ABC):
    
    """A policy that determines which downstream server to send a request to
    based on information from the DHT"""

    def __init__(self, model: MultiTaskModel, dht: DistributedHashTable, update_interval: int):
        self.model = model
        self.dht = dht
        self.update_interval = update_interval
        self.last_update = 0
        self.update_if_needed()

    @abstractmethod
    def route(self, request: InferRequest) -> int:
        pass
    
    def _update(self):
        pass

    def update_if_needed(self):
        current_time = time.time()
        if current_time - self.last_update > self.update_interval:
            self._update()
        self.last_update = time.time()


"""Server-side routing policies"""


class LoadBasedRoutingPolicy(RoutingPolicy):

    """Chooses a downstream server to send a request to based on
    the load of the server (estimated by queue length)"""

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


class RandomRoutingPolicy(RoutingPolicy):
    
    """Randomly chooses a downstream server to send a request to
    based on information from the DHT"""

    def route(self, request: InferRequest) -> int:
        # Get all servers serving a given stage
        next_stage = self.model.get_stage(request.task_name, request.next_stage_idx)
        possible_servers = self.dht.get_servers_with_stage(next_stage.name)

        if len(possible_servers) > 0:
            # Return random server in the list
            return random.choice(possible_servers)
        else:
            return -1


"""Client-side routing policies"""


class ServerSelectionPolicy(RoutingPolicy):
    
    """Choose the server that can serve the first stage of the new request. 
    Return the index of the best server"""

    def route(self, request: InferRequest) -> int:
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
