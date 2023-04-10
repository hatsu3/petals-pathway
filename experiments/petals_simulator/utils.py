import random

import networkx as nx
import matplotlib.pyplot as plt

import torch.nn as nn
# from transformers.models.bloom.modeling_bloom import BloomBlock, BloomConfig

from multitask_model import MultiTaskModel, Stage, TaskPath
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
    if not nx.is_directed_acyclic_graph(G):
        return generate_random_dag(num_nodes, num_edges)

    # Check if there is any node with no incoming edges and no outgoing edges
    for node in G.nodes():
        if G.in_degree(node) == 0 and G.out_degree(node) == 0:
            return generate_random_dag(num_nodes, num_edges)

    return G


def visualize_dag(G):
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=20, font_weight='bold', arrowsize=30)
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
            paths.extend(list(nx.all_simple_paths(G, source_node, sink_node)))

    # create a TaskPath for each path
    tasks = list()
    for i, path in enumerate(paths):
        task = TaskPath(f"task{i}", [stages[node_idx] for node_idx in path])
        tasks.append(task)
    
    return MultiTaskModel(stages, tasks)


# generate fake profiling results (each stage is exactly the same and takes 10ms)
def get_dummy_model_and_prof_results(num_nodes=100, num_edges=250):
    random_graph = generate_random_dag(num_nodes, num_edges)
    model: MultiTaskModel = dag_to_multitask_model(random_graph)
    prof_results = ProfilingResults(
        results=[
            {
                "name": stage_name,
                "type": DummyStage.__name__,
                "batch_size": 1,
                "latency_ms": 10,
            } 
            for stage_name, stage in model.stages.items()
        ],
        batch_sizes=[1],
        prof_kwargs={},
    )
    return model, prof_results


if __name__ == "__main__":
    get_dummy_model_and_prof_results(10, 10)
