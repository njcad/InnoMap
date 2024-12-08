import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import train_test_split


def neighbor_sample_subgraph(data, node_indices, num_hops=2, num_neighbors=10):
    """
    Given a set of node indices, sample their neighborhood to form a subgraph.
    """
    # NeighborLoader returns mini-batches for each seed node set.
    # If we set batch_size = len(node_indices), we get one batch containing all these nodes.
    loader = NeighborLoader(
        data,
        num_neighbors=[num_neighbors]*num_hops,
        input_nodes=node_indices,
        batch_size=len(node_indices),
        shuffle=False
    )

    # Load one full batch
    sampled_data = next(iter(loader))
    # sampled_data is a Data object containing the sampled subgraph
    # The 'sampled_data' node indices are mapped to [0, ...], so it's a clean subgraph.

    return sampled_data


def load_and_neighbor_split(path, test_size=0.2, random_state=42, num_hops=2, num_neighbors=10):
    # Load the full graph
    data = torch.load(path)

    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)

    # Split nodes into train and test sets
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )

    # Convert to tensors
    train_indices_t = torch.tensor(train_indices, dtype=torch.long)
    test_indices_t = torch.tensor(test_indices, dtype=torch.long)

    # Sample subgraphs around these nodes
    train_graph = neighbor_sample_subgraph(data, train_indices_t, num_hops=num_hops, num_neighbors=num_neighbors)
    test_graph = neighbor_sample_subgraph(data, test_indices_t, num_hops=num_hops, num_neighbors=num_neighbors)

    # Save the resulting subgraphs
    torch.save(train_graph, "./graphs/train_graph.pt")
    torch.save(test_graph, "./graphs/test_graph.pt")

    return train_graph, test_graph


if __name__ == "__main__":
    graph_path = "./graphs/full.pt"
    train_graph, test_graph = load_and_neighbor_split(graph_path)
    print(train_graph)
    print(test_graph)
