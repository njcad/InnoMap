# src/model.py


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # first two conv layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        # MLP layer for link prediction
        self.link_predictor = torch.nn.Sequential(
            torch.nn.Linear(in_channels + out_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid()
        )

    def get_embeddings(self, x, edge_index):
        """Get node embeddings from the GCN."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def predict_links(self, source_emb, target_emb):
        """Predict links given source and target embeddings."""
        edge_emb = torch.cat([source_emb, target_emb], dim=1)
        return self.link_predictor(edge_emb)

    def forward(self, x, edge_index, node_pairs):
        # Grab the target embeddings FIRST (as the LLM embeddings)
        target_emb = x[node_pairs[:, 1]]

        # Get embeddings using GCN
        x = self.get_embeddings(x, edge_index)

        # Extract embeddings for node pairs
        source_emb = x[node_pairs[:, 0]]

        # Predict link probability
        link_probs = self.predict_links(source_emb, target_emb)
        return link_probs







### NOTE: the below code may be functional, however, I (nate) am just commenting it out for now to test above model.

# class GCNEncoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCNEncoder, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return x  # Returns z_i

# #THIS IS WRONG SINCE IT SHOULD TAKE INCHANNELS_textencoder + INCHANNELS_gcnoutput!!!!!!
# class EdgePredictor(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super(EdgePredictor, self).__init__()
#         self.fc1 = torch.nn.Linear(2 * in_channels, hidden_channels)
#         self.fc2 = torch.nn.Linear(hidden_channels, 1)

#     def forward(self, src_feature, x_new):
#         x = torch.cat([src_feature, x_new], dim=1)
#         x = F.relu(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))
#         return x.squeeze()

# class GCNLinkPredictor(torch.nn.Module):
#     def __init__(self, encoder, predictor):
#         super(GCNLinkPredictor, self).__init__()
#         self.encoder = encoder
#         self.predictor = predictor

#     def forward(self, data, src_indices, x_new):
#         z = self.encoder(data.x, data.edge_index)
#         src_embeddings = z[src_indices]
#         return self.predictor(src_embeddings, x_new)
