import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

file_path = './graphs/first_10_5.pt'
# Load the saved graph data
data = torch.load(file_path)

print(data)  # Prints a summary of the graph

# For more detailed inspection, check each element's dimensions
print("Node feature matrix shape (x):", data.x.shape if data.x is not None else "No node features")
print("Edge index shape:", data.edge_index.shape)
print("Edge attributes shape:", data.edge_attr.shape if data.edge_attr is not None else "No edge attributes")
print("Labels shape (y):", data.y.shape if data.y is not None else "No labels")
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)

# Convert to NetworkX graph for plotting
G = to_networkx(data, to_undirected=False)

# Plot the graph
plt.figure(figsize=(12, 12))
nx.draw_networkx(
    G,
    node_size=20,
    with_labels=False,
    edge_color='gray',
    alpha=0.7
)
plt.title("Directed Graph from Papers and References")
plt.show()
