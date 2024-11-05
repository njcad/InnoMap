# src/model.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv




class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # Returns z_i

#THIS IS WRONG SINCE IT SHOULD TAKE INCHANNELS_textencoder + INCHANNELS_gcnoutput!!!!!!
class EdgePredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(EdgePredictor, self).__init__()
        self.fc1 = torch.nn.Linear(2 * in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, src_feature, x_new):
        x = torch.cat([src_feature, x_new], dim=1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()

class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, encoder, predictor):
        super(GCNLinkPredictor, self).__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, data, src_indices, x_new):
        z = self.encoder(data.x, data.edge_index)
        src_embeddings = z[src_indices]
        return self.predictor(src_embeddings, x_new)
