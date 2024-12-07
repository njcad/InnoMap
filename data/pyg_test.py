from torch_geometric.datasets import Planetoid
from torch_geometric.loader import ClusterData

dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(dataset)
cluster_data = ClusterData(dataset[0], num_parts=50)
print(cluster_data)