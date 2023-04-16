import os
import random
import logging
import threading

from server import Server
from client import Client, RequestMode
from dht import DistributedHashTable
from latency_estimator import LatencyEstimator, generate_random_location
from utils import get_dummy_model_and_prof_results, TraceFile
from scheduling import RandomSchedulingPolicy, LatencyAwareSchedulingPolicy
from routing import (
    RoutingPolicy, RandomRoutingPolicy, 
    QueueLengthRoutingPolicy, RequestRateRoutingPolicy
)
from stage_assignment import (
    StageAssignmentPolicy, AllToAllStageAssignmentPolicy, 
    UniformStageAssignmentPolicy, RequestRateStageAssignmentPolicy
)


class Simulator:
    def __init__(self, servers: list[Server], clients: list[Client]):
        self.servers = servers
        self.clients = clients

    def run(self, server_run_time: int = 20, client_run_time: int = 20):
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


def run_simulation(
        num_servers: int = 8, num_clients: int = 50,
        server_run_time: int = 20, client_run_time: int = 20,
        stage_latency: int = 30, model_gen_seed: int = 42,
        sched_policy_cls: type = LatencyAwareSchedulingPolicy,
        routing_policy_cls: type = RandomRoutingPolicy,
        stage_assign_policy_cls: type = RequestRateStageAssignmentPolicy,
        server_announce_interval: int = 3,
        server_rebalance_interval: int = 3,
        client_request_mode: RequestMode = RequestMode.POISSON,
        client_request_avg_interval: int = 1,
    ):
    # Construct dummy model, initialize DHT and get latency estimates
    # we specify a seed to make the results reproducible
    model, prof_results = get_dummy_model_and_prof_results(
        stage_latency=stage_latency, 
        seed=model_gen_seed
    )
    dht = DistributedHashTable(model)
    latency_est = LatencyEstimator.load("data/latency_estimator.pkl")

    # Set the policies
    sched_policy = sched_policy_cls(model, prof_results)
    # TODO: remove update_interval because it is not used now
    routing_policy = routing_policy_cls(model, dht, update_interval=3)
    stage_assign_policy = stage_assign_policy_cls(model, dht)

    servers = list()
    server_port_offset = random.randint(5000, 6000)
    for i in range(num_servers):
        servers.append(Server(
            ip="127.0.0.1",
            port=server_port_offset + i,
            location=generate_random_location(),
            dht=dht,
            model=model,
            prof_results=prof_results,
            latency_est=latency_est,
            num_router_threads=1,
            announce_interval=server_announce_interval,
            rebalance_interval=server_rebalance_interval,
            sched_policy=sched_policy,
            routing_policy=routing_policy,
            stage_assignment_policy=stage_assign_policy,
        ))

    clients = list()
    client_port_offset = random.randint(7000, 8000)
    for i in range(num_clients):
        clients.append(Client(
            ip="127.0.0.1",
            port=client_port_offset + i,
            client_id=i,
            location=generate_random_location(),
            task_name=random.choice(list(model.paths.keys())),
            dht=dht,
            model=model,
            latency_est=latency_est,
            server_sel_policy=routing_policy,
            request_mode=client_request_mode,
            request_avg_interval=client_request_avg_interval,
        ))

    simulator = Simulator(servers, clients)
    simulator.run(
        server_run_time=server_run_time,
        client_run_time=client_run_time,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with TraceFile('trace.json'):
        run_simulation()
