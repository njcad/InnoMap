import os
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def sample_subset_negative_edges(graph):
    """
    Samples a subset of negative edges for the graph.

    Parameters:
    - graph: PyTorch Geometric Data object.
    - num_neg_samples: Number of negative edges to sample.

    Returns:
    - negative_edges: Tensor of sampled negative edges.
    """
    num_nodes = graph.num_nodes
    edge_set = set((u.item(), v.item()) for u, v in graph.edge_index.t())

    size = graph.edge_index.size(1)
    negative_edges = []
    while len(negative_edges) < size:
        uv = torch.randint(0, num_nodes, (size, 2))
        uv = uv[uv[:, 0] != uv[:, 1]]  # Remove self-loops
        for u, v in uv.tolist():
            if (u, v) not in edge_set and (v, u) not in edge_set:
                negative_edges.append((u, v))
                if len(negative_edges) >= size:
                    break

    return torch.tensor(negative_edges, dtype=torch.long)


def save_negative_edges(graph, output_file):
    """
    Generates and saves a subset of negative edges for a graph.

    Parameters:
    - graph: PyTorch Geometric Data object.
    - output_file: Path to save the negative edges.
    - num_neg_samples: Number of negative edges to sample.
    """
    print(f"Sampling {graph.edge_index.size(1)} negative edges for a graph with {graph.num_nodes} nodes...")
    negative_edges = sample_subset_negative_edges(graph)
    print(f"Total negative edges sampled: {negative_edges.size(0)}")

    # Save the negative edges to a file
    torch.save(negative_edges, output_file)
    print(f"Negative edges saved to {output_file}")


if __name__ == "__main__":
    GRAPH_PATH = "../data/graphs/eval_graph.pt"  # Path to the input graph
    OUTPUT_PATH = "../data/dataset/full_negative_edges.pt"  # Path to save negative edges
    NUM_NEG_SAMPLES = 100000  # Number of negative edges to sample

    # Load the graph
    print(f"Loading graph from {GRAPH_PATH}...")
    graph = torch.load(GRAPH_PATH)

    # Generate and save negative edges
    save_negative_edges(graph, OUTPUT_PATH)
