import torch
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData
from sklearn.model_selection import train_test_split
import numpy as np

def load_and_split_graph(path, test_size=0.2, random_state=42, num_parts=50):
    # Load the full graph
    graph = torch.load(path, )

    # Use ClusterData to partition the graph into connected clusters.
    # This helps ensure that the train/test subgraphs are more "connected"
    # than a random node-level split.
    cluster_data = ClusterData(graph, num_parts=num_parts)
    # cluster_data.perm is a permutation of all nodes
    # cluster_data.partptr defines the boundaries of clusters in `perm`
    # For cluster i, the nodes are cluster_data.perm[partptr[i]:partptr[i+1]]

    partptr = cluster_data.partptr
    perm = cluster_data.perm
    num_clusters = partptr.size(0) - 1  # number of clusters

    # Create cluster indices
    cluster_indices = np.arange(num_clusters)

    # Perform a cluster-level train/test split
    # This picks certain clusters for training and others for testing.
    train_clusters, test_clusters = train_test_split(
        cluster_indices,
        test_size=test_size,
        random_state=random_state
    )

    # Get the nodes belonging to these clusters
    train_nodes = []
    for c in train_clusters:
        start, end = partptr[c].item(), partptr[c+1].item()
        train_nodes.append(perm[start:end])
    train_nodes = torch.cat(train_nodes)

    test_nodes = []
    for c in test_clusters:
        start, end = partptr[c].item(), partptr[c+1].item()
        test_nodes.append(perm[start:end])
    test_nodes = torch.cat(test_nodes)

    # Create masks
    num_nodes = graph.x.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_nodes] = True
    test_mask[test_nodes] = True

    graph.train_mask = train_mask
    graph.test_mask = test_mask

    # Filter edges for the train graph
    train_edge_mask = train_mask[graph.edge_index[0]] & train_mask[graph.edge_index[1]]
    train_edge_index = graph.edge_index[:, train_edge_mask]

    # Filter edges for the test graph
    test_edge_mask = test_mask[graph.edge_index[0]] & test_mask[graph.edge_index[1]]
    test_edge_index = graph.edge_index[:, test_edge_mask]

    # Renumber nodes for each subgraph
    train_graph = Data(
        x=graph.x[train_mask],
        edge_index=renumber_edge_index(train_edge_index, train_nodes)
    )

    test_graph = Data(
        x=graph.x[test_mask],
        edge_index=renumber_edge_index(test_edge_index, test_nodes)
    )

    # Save the subgraphs
    torch.save(train_graph, "./graphs/train_graph.pt")
    torch.save(test_graph, "./graphs/test_graph.pt")

    return graph

def renumber_edge_index(edge_index, node_subset):
    # Map old node IDs to new IDs (0 to len(node_subset)-1)
    node_subset_sorted, _ = torch.sort(node_subset)
    mapping = torch.full((node_subset_sorted[-1] + 1,), -1, dtype=torch.long)
    mapping[node_subset_sorted] = torch.arange(node_subset_sorted.size(0))
    new_edge_index = mapping[edge_index]
    return new_edge_index

if __name__ == "__main__":
    # Example usage
    graph_path = "./graphs/full.pt"
    split_graph = load_and_split_graph(graph_path)
    print(f"Total nodes: {split_graph.x.size(0)}")
    print(f"Training nodes: {split_graph.train_mask.sum().item()}")
    print(f"Test nodes: {split_graph.test_mask.sum().item()}")
