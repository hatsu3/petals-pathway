import random
import threading
import time

from server import RoutingPolicy, SchedulingPolicy, Server, StageAssignmentPolicy
from client import AsyncClient, RequestMode, ServerSelectionPolicy
from dht import DistributedHashTable
from stage_profiler import StageProfiler
from multitask_model import MultiTaskModel, Stage
from latency_estimator import LatencyEstimator, generate_random_location
from utils import get_dummy_model_and_prof_results


class Simulator:
    def __init__(self, servers: list[Server], clients: list[AsyncClient]):
        self.servers = servers
        self.clients = clients

    def run(self):
        # Start the server thread
        server_threads = [
            threading.Thread(target=server.run, args=(30,))  # run for 30 seconds
            for server in self.servers
        ]
        for server_thread in server_threads:
            server_thread.start()

        # Start the client threads
        client_threads = [
            threading.Thread(target=client.run)  # TODO: stop client after a certain time
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
    sched_policy = SchedulingPolicy(model)
    routing_policy = RoutingPolicy(model, dht, update_interval=3)
    stage_assign_policy = StageAssignmentPolicy(model, dht)

    num_servers = 10
    num_clients = 10

    servers = list()
    for i in range(num_servers):
        servers.append(Server(
            ip="127.0.0.1",
            port=8000 + i,
            server_id=i,
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
        clients.append(AsyncClient(
            ip="127.0.0.1",
            send_port=9000 + i,
            recv_port=10000 + i,
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
