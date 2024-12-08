import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def read_predictions(results_file):
    """
    Reads predictions from the results file and extracts edge predictions.
    
    Parameters:
    - results_file: Path to the file containing edge predictions.
    
    Returns:
    - edge_predictions: A list of tuples (source, target, probability).
    """
    edge_predictions = []
    with open(results_file, 'r') as f:
        for line in f:
            if "Node" in line:
                parts = line.strip().split(":")
                node_pair = tuple(map(int, parts[1].split(",")))
            elif "Probability" in line:
                prob = float(line.split(":")[1].strip())
                edge_predictions.append((*node_pair, prob))
    return edge_predictions

def visualize_predictions_from_file(graph, predictions_file, top_k=10, node_labels=None, save_path=None):
    """
    Visualizes the graph with predicted edges read directly from a results file.
    
    Parameters:
    - graph: A PyTorch Geometric Data object.
    - predictions_file: Path to the file containing edge predictions.
    - top_k: Number of top edges (by probability) to visualize.
    - node_labels: Optional dictionary mapping node indices to labels for display.
    - save_path: Optional path to save the resulting visualization.
    """
    # Read predictions from file
    edge_predictions = read_predictions(predictions_file)
    
    # Sort and select top-k edges
    edge_predictions = sorted(edge_predictions, key=lambda x: x[2], reverse=True)[:top_k]
    predicted_edges = [(u, v) for u, v, _ in edge_predictions]
    predicted_probs = [p for _, _, p in edge_predictions]

    # Create a networkx graph from the PyTorch Geometric graph
    G = nx.DiGraph() if graph.is_directed() else nx.Graph()
    edge_index = graph.edge_index.t().tolist()
    G.add_edges_from(edge_index)
    
    # Create a plot
    pos = nx.spring_layout(G, seed=42)  # Layout for nodes
    plt.figure(figsize=(12, 8))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue', alpha=0.8)

    # Draw existing edges in the graph
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)

    # Highlight predicted edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=predicted_edges,
        edge_color='red',
        width=2,
        alpha=0.8,
        connectionstyle="arc3,rad=0.2"
    )

    # Add labels (optional)
    if node_labels:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

    # Add a colorbar for predicted probabilities
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(predicted_probs), vmax=max(predicted_probs)))
    sm.set_array(predicted_probs)
    plt.colorbar(sm, label="Prediction Probability")

    # Title and save
    plt.title(f"Graph Visualization with Top {top_k} Predicted Edges")
    if save_path:
        plt.savefig(save_path, format="PNG", dpi=300)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load the graph from the results directory
    graph_file = "../data/graphs/eval_graph.pt"
    predictions_file = "./results/new_text_predictions.txt"  # Adjust if the file name differs
    save_path = "./results/graph_visualization.png"

    graph = torch.load(graph_file)  # Load the PyTorch Geometric graph
    node_labels = {i: f"Node {i}" for i in range(graph.num_nodes)}  # Optional: Create node labels
    
    # Visualize predictions
    visualize_predictions_from_file(
        graph=graph,
        predictions_file=predictions_file,
        top_k=10,
        node_labels=node_labels,
        save_path=save_path
    )
