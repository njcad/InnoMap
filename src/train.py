# src/train.py
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# import our model from model.py
from model import GCNLinkPredictor


def sample_negative_edges(num_nodes, edge_index, num_samples):
    """
    Sample random negative edges (non-existent links).
    :param num_nodes: Number of nodes in the graph.
    :param edge_index: Tensor of existing edges.
    :param num_samples: Number of negative edges to sample.
    :return: Tensor of negative edges.
    """
    edge_set = set([tuple(edge) for edge in edge_index.t().tolist()])
    negative_edges = []
    while len(negative_edges) < num_samples:
        u, v = torch.randint(0, num_nodes, (2,))
        if (u.item(), v.item()) not in edge_set and (v.item(), u.item()) not in edge_set:
            negative_edges.append((u.item(), v.item()))
    return torch.tensor(negative_edges, dtype=torch.long)


def train(model, data, optimizer, num_epochs):
    """
    Train the GCN model for link prediction.
    :param model: GCN model instance.
    :param data: PyTorch Geometric Data object.
    :param optimizer: Optimizer for training.
    :param num_epochs: Number of training epochs.
    """
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Sample positive and negative edges
        pos_edges = data.edge_index.t()  # Positive edges
        neg_edges = sample_negative_edges(data.num_nodes, data.edge_index, pos_edges.size(0))

        # Combine edges and labels
        all_edges = torch.cat([pos_edges, neg_edges], dim=0)
        labels = torch.cat([torch.ones(pos_edges.size(0)), torch.zeros(neg_edges.size(0))])

        # Forward pass and loss computation
        link_probs = model(data.x, data.edge_index, all_edges)
        loss = F.binary_cross_entropy(link_probs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Evaluate on training set
        with torch.no_grad():
            auc = roc_auc_score(labels.cpu(), link_probs.cpu())
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, AUC: {auc:.4f}")


# === INFERENCE ===
def predict_links(model, data, new_paper_embedding=None):
    """
    Predict links for a new paper or existing graph.
    :param model: Trained GCN model.
    :param data: PyTorch Geometric Data object.
    :param new_paper_embedding: Embedding of the new paper (optional).
    :return: Predicted link probabilities for all candidate edges.
    """
    model.eval()

    if new_paper_embedding is not None:
        # Add new paper embedding as a new node
        new_embedding = torch.tensor(new_paper_embedding, dtype=torch.float).unsqueeze(0)
        data.x = torch.cat([data.x, new_embedding], dim=0)
        new_node_idx = data.x.size(0) - 1
        candidate_edges = torch.tensor([[new_node_idx, i] for i in range(new_node_idx)], dtype=torch.long)
    else:
        # Predict links for all pairs in the existing graph
        candidate_edges = torch.cartesian_prod(torch.arange(data.num_nodes), torch.arange(data.num_nodes))

    with torch.no_grad():
        link_probs = model(data.x, data.edge_index, candidate_edges)
    return link_probs



def main():
    """
    Main script to actually run the thing.
    """
    










### NOTE: same story, commented out code below might work, but just testing this stuff


# import torch
# import torch.nn.functional as F
# from torch.optim import Adam
# from tqdm import tqdm

# from dataset import get_dataloader
# from model import GCNEncoder, EdgePredictor, GCNLinkPredictor

# def train_model(config):
#     # Configuration parameters
#     data_path = config['data_path']
#     batch_size = config['batch_size']
#     device = torch.device(config['device'])
#     num_epochs = config['num_epochs']
#     learning_rate = config['learning_rate']
#     encoder_hidden_channels = config['encoder_hidden_channels']
#     encoder_out_channels = config['encoder_out_channels']
#     predictor_hidden_channels = config['predictor_hidden_channels']
    
#     # Initialize dataloader
#     train_loader = get_dataloader(
#         data_path=data_path,
#         batch_size=batch_size,
#         num_negative_samples=config.get('num_negative_samples', 1),
#         mode='train'
#     )
#     data = train_loader.dataset.data.to(device)
    
#     # Initialize model
#     encoder = GCNEncoder(
#         in_channels=data.x.size(-1),
#         hidden_channels=encoder_hidden_channels,
#         out_channels=encoder_out_channels
#     )
#     predictor = EdgePredictor(
#         in_channels=encoder_out_channels,
#         hidden_channels=predictor_hidden_channels
#     )
#     model = GCNLinkPredictor(encoder, predictor).to(device)
    
#     # Initialize optimizer
#     optimizer = Adam(model.parameters(), lr=learning_rate)
    
#     # Training loop
#     for epoch in range(1, num_epochs + 1):
#         model.train()
#         total_loss = 0
#         for src_features, x_news, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'):
#             src_features = src_features.to(device)
#             x_news = x_news.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()
#             # Forward pass
#             outputs = model.predictor(src_features, x_news)
#             # Compute loss
#             loss = F.binary_cross_entropy(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() * labels.size(0)
#         avg_loss = total_loss / len(train_loader.dataset)
#         print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
        
#         # Optionally, evaluate on validation set
#         # if epoch % config.get('eval_interval', 5) == 0:
#         #     val_auc = evaluate_model(model, val_loader, device)
#         #     print(f'Validation AUC: {val_auc:.4f}')
    
#     # Save the trained model
#     model_save_path = config.get('model_save_path', './Models/gcn_link_predictor.pth')
#     torch.save(model.state_dict(), model_save_path)
#     print(f'Model saved to {model_save_path}')
    
#     return model

# def evaluate_model(model, dataloader, device):
#     from sklearn.metrics import roc_auc_score
#     model.eval()
#     preds = []
#     labels = []
#     with torch.no_grad():
#         for src_features, x_news, batch_labels in dataloader:
#             src_features = src_features.to(device)
#             x_news = x_news.to(device)
#             outputs = model.predictor(src_features, x_news)
#             preds.append(outputs.cpu())
#             labels.append(batch_labels)
#     preds = torch.cat(preds)
#     labels = torch.cat(labels)
#     auc = roc_auc_score(labels.numpy(), preds.numpy())
#     return auc

# if __name__ == '__main__':
#     # Load configuration parameters
#     import yaml
#     with open('./configs/defaults.yaml', 'r') as f:
#         config = yaml.safe_load(f)
#     # Start training
#     trained_model = train_model(config)
