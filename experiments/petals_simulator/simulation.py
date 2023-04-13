import random
import threading
import time

from server import RoutingPolicy, SchedulingPolicy, SchedulingEstimationPolicy, Server, StageAssignmentPolicy, BaselineStageAssignmentPolicy
from client import Client, RequestMode, ServerSelectionPolicy
from dht import DistributedHashTable
from stage_profiler import StageProfiler
from multitask_model import MultiTaskModel, Stage
from latency_estimator import LatencyEstimator, generate_random_location
from utils import get_dummy_model_and_prof_results

import logging


class Simulator:
    def __init__(self, servers: list[Server], clients: list[Client]):
        self.servers = servers
        self.clients = clients

    def run(self, server_run_time: int = 30, client_run_time: int = 30):
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
    dht = DistributedHashTable()
    model, prof_results = get_dummy_model_and_prof_results()
    latency_est = LatencyEstimator.load("data/latency_estimator.pkl")

    server_sel_policy = ServerSelectionPolicy(model, dht)
    sched_policy = SchedulingEstimationPolicy(model, prof_results)
    routing_policy = RoutingPolicy(model, dht, update_interval=3)
    stage_assign_policy = BaselineStageAssignmentPolicy(model, dht)

    num_servers = 8
    num_clients = 50

    servers = list()
    for i in range(num_servers):
        servers.append(Server(
            ip="127.0.0.1",
            port=15000 + i,
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
        # time.sleep(2.0)

    clients = list()
    for i in range(num_clients):
        clients.append(Client(
            ip="127.0.0.1",
            port=25000 + i,
            client_id=i,
            location=generate_random_location(),
            task_name=random.choice(list(model.paths.keys())),
            dht=dht,
            model=model,
            latency_est=latency_est,
            server_sel_policy=server_sel_policy,
            request_mode=RequestMode.POISSON,
            request_avg_interval=1,
            update_interval=5
        ))

    simulator = Simulator(servers, clients)
    simulator.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with open('trace.json', 'a') as f:
        f.write(f"[\n")
    run_simulation()
    with open('trace.json', 'a') as f:
        f.write(f"]\n")
