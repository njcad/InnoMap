import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

file_path = './graphs/first_10_5.pt'
# Load the saved graph data
data = torch.load(file_path)

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
