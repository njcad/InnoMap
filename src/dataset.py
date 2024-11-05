# src/dataset.py

import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes
from torch.utils.data import Dataset, DataLoader
import random

class CitationDataset(Dataset):
    def __init__(self, data_path, num_negative_samples=1, mode='train'):
        self.data = torch.load(data_path)
        self.mode = mode
        self.num_negative_samples = num_negative_samples
        self.prepare_data()

    def prepare_data(self):
        data = self.data
        # For training, we simulate the new node by randomly removing nodes
        if self.mode == 'train':
            # Randomly select nodes to be treated as new nodes
            num_nodes = data.num_nodes
            num_new_nodes = int(0.1 * num_nodes)  # 10% as new nodes
            self.new_node_indices = random.sample(range(num_nodes), num_new_nodes)
            self.existing_node_indices = list(set(range(num_nodes)) - set(self.new_node_indices))

            # Create subgraph without the new nodes
            mask = torch.ones(num_nodes, dtype=torch.bool)
            mask[self.new_node_indices] = False
            data.edge_index, data.edge_attr = remove_isolated_nodes(
                data.edge_index[:, mask[data.edge_index[0]] & mask[data.edge_index[1]]],
                num_nodes=num_nodes
            )
            data.x = data.x[mask]
            self.data = data

            # Prepare positive and negative samples
            self.pairs = []
            for new_node_idx in self.new_node_indices:
                x_new = self.data.x[new_node_idx]
                # Positive samples: nodes that have edges to the new node
                existing_edges = data.edge_index[:, data.edge_index[1] == new_node_idx]
                positive_indices = existing_edges[0]
                for pos_idx in positive_indices:
                    self.pairs.append((pos_idx, x_new, 1))
                # Negative samples: Randomly sample nodes that don't have edges to the new node
                negative_indices = random.sample(self.existing_node_indices, self.num_negative_samples)
                for neg_idx in negative_indices:
                    self.pairs.append((neg_idx, x_new, 0))
        else:
            # For validation and testing, we can prepare data similarly or use predefined splits
            pass  # Implement as needed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_idx, x_new, label = self.pairs[idx]
        src_feature = self.data.x[src_idx]
        return src_feature, x_new, label

def collate_fn(batch):
    src_features, x_news, labels = zip(*batch)
    src_features = torch.stack(src_features)
    x_news = torch.stack(x_news)
    labels = torch.tensor(labels, dtype=torch.float)
    return src_features, x_news, labels

def get_dataloader(data_path, batch_size=64, num_negative_samples=1, mode='train'):
    dataset = CitationDataset(
        data_path=data_path,
        num_negative_samples=num_negative_samples,
        mode=mode
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        collate_fn=collate_fn
    )
    return dataloader
