import random

from .server import Server
from .client import AsyncClient
from .dht import DistributedHashTable


class Simulator:
    def __init__(self, num_servers: int, num_clients: int, num_requests: int):
        # set up servers (join one by one)
        # set up clients (send requests in some pattern)
        # run for a while and dump stats
        # clean up and exit gracefully
        self.num_servers = num_servers
        self.num_clients = num_clients
        self.num_requests = num_requests

        self.dht = DistributedHashTable()

        self.servers = []
        for i in range(num_servers):
            self.servers.append(Server(self.dht, i))

        self.clients = []
        for i in range(num_clients):
            self.clients.append(AsyncClient(self.dht, i))

    def run(self):
        # start servers
        for server in self.servers:
            server.start()

        # start clients
        for client in self.clients:
            client.start()
