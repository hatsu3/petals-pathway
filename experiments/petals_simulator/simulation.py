import os
import random
import threading

from server import Server
from client import Client, RequestMode, ServerSelectionPolicy
from dht import DistributedHashTable
from latency_estimator import LatencyEstimator, generate_random_location
from utils import get_dummy_model_and_prof_results
from scheduling import RandomSchedulingPolicy, LatencyAwareSchedulingPolicy
from routing import RoutingPolicy, RandomRoutingPolicy
from stage_assignment import (
    StageAssignmentPolicy, AllToAllStageAssignmentPolicy, 
    UniformStageAssignmentPolicy, RequestRateStageAssignmentPolicy
)

import logging


class Simulator:
    def __init__(self, servers: list[Server], clients: list[Client]):
        self.servers = servers
        self.clients = clients

    def run(self, server_run_time: int = 22, client_run_time: int = 20):
        # Start the server thread
        server_threads = [
            threading.Thread(target=server.run, args=(server_run_time,))
            for server in self.servers
        ]
        for server_thread in server_threads:
            server_thread.start()

        # Start the client threads
        client_threads = [
            threading.Thread(target=client.run, args=(client_run_time,))
            for client in self.clients
        ]
        for client_thread in client_threads:
            client_thread.start()

        # Wait for the server and client threads to finish
        for server_thread in server_threads:
            server_thread.join()
        
        for client_thread in client_threads:
            client_thread.join()


def run_simulation():
    # Construct dummy model, initialize DHT and get latency estimates
    model, prof_results = get_dummy_model_and_prof_results(stage_latency=30)
    dht = DistributedHashTable(model)
    latency_est = LatencyEstimator.load("data/latency_estimator.pkl")

    # Set the policies
    server_sel_policy = ServerSelectionPolicy(model, dht, update_interval=3)
    sched_policy = LatencyAwareSchedulingPolicy(model, prof_results)
    routing_policy = RandomRoutingPolicy(model, dht, update_interval=3)
    stage_assign_policy = RequestRateStageAssignmentPolicy(model, dht)

    # Set low number of servers and clients, for testing
    num_servers = 8
    num_clients = 50

    servers = list()
    for i in range(num_servers):
        servers.append(Server(
            ip="127.0.0.1",
            port=random.randint(10000, 20000),
            location=generate_random_location(),
            dht=dht,
            model=model,
            prof_results=prof_results,
            latency_est=latency_est,
            num_router_threads=1,
            announce_interval=3,
            rebalance_interval=5,
            sched_policy=sched_policy,
            routing_policy=routing_policy,
            stage_assignment_policy=stage_assign_policy,
        ))

    clients = list()
    for i in range(num_clients):
        clients.append(Client(
            ip="127.0.0.1",
            port=random.randint(10000, 20000),
            client_id=i,
            location=generate_random_location(),
            task_name=random.choice(list(model.paths.keys())),
            dht=dht,
            model=model,
            latency_est=latency_est,
            server_sel_policy=server_sel_policy,
            request_mode=RequestMode.POISSON,
            request_avg_interval=1
        ))

    simulator = Simulator(servers, clients)
    simulator.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if os.path.exists('trace.json'):
        os.remove('trace.json')
    with open('trace.json', 'a') as f:
        f.write(f"[\n")
    run_simulation()
    with open('trace.json', 'a') as f:
        f.write(f"]\n")
