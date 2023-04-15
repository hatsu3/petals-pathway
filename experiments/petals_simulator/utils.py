import random
import threading
import time

import networkx as nx
import matplotlib.pyplot as plt

import torch.nn as nn
# from transformers.models.bloom.modeling_bloom import BloomBlock, BloomConfig

from multitask_model import MultiTaskModel, Stage, TaskPath
from messages import InferRequest
from trace_visualizer import TraceVisualizer
from stage_profiler import ProfilingResults


""" create a single bloom block
config = BloomConfig.from_pretrained("bloom")
block = BloomBlock(config) # type: ignore

num_heads = config.n_head
hidden_size = config.hidden_size
seq_length = 2048

hidden_states: batch_size, seq_length, hidden_size
attention_mask: batch_size, seq_length
alibi: batch_size * num_heads, 1, seq_length"""


def generate_random_dag(num_nodes, num_edges):
    G = nx.DiGraph()
    G.add_nodes_from([(i, dict(name=f"stage{i}")) for i in range(num_nodes)])

    edges = [(u, v) for u in range(num_nodes) for v in range(u+1, num_nodes)]
    random.shuffle(edges)
    G.add_edges_from(edges[:num_edges])

    # Check if the generated graph is a DAG
    if not nx.is_directed_acyclic_graph(G): # type: ignore
        return generate_random_dag(num_nodes, num_edges)

    # Check if there is any node with no incoming edges and no outgoing edges
    for node in G.nodes():
        if G.in_degree(node) == 0 and G.out_degree(node) == 0:
            return generate_random_dag(num_nodes, num_edges)

    return G


def visualize_dag(G):
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=20, font_weight='bold', arrowsize=30) # type: ignore
    plt.savefig('dag.png')


class DummyStage(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# convert the DAG to a MultiTaskModel
def dag_to_multitask_model(G: nx.DiGraph):
    stages = [
        Stage(
            name=node[1]["name"],
            module_cls=DummyStage,
        )
        for node in G.nodes(data=True)
    ]
    
    # find all paths from source nodes to sink nodes in the DAG
    source_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]
    sink_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]
    
    paths = list()
    for source_node in source_nodes:
        for sink_node in sink_nodes:
            paths.extend(list(nx.all_simple_paths(G, source_node, sink_node))) # type: ignore

    # create a TaskPath for each path
    tasks = list()
    for i, path in enumerate(paths):
        task = TaskPath(f"task{i}", [stages[node_idx] for node_idx in path])
        tasks.append(task)
    
    return MultiTaskModel(stages, tasks)


# generate fake profiling results (each stage is exactly the same and takes 10ms)
def get_dummy_model_and_prof_results(num_nodes=100, num_edges=250, stage_latency=10):
    random_graph = generate_random_dag(num_nodes, num_edges)
    model: MultiTaskModel = dag_to_multitask_model(random_graph)
    prof_results = ProfilingResults(
        results=[
            {
                "name": stage_name,
                "type": DummyStage.__name__,
                "batch_size": 1,
                "latency_ms": stage_latency,
            } 
            for stage_name, stage in model.stages.items()
        ],
        batch_sizes=[1],
        prof_kwargs={},
    )
    return model, prof_results


""" Utils for simulating execution of a stage on GPU """

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


if __name__ == "__main__":
    get_dummy_model_and_prof_results(10, 10)
