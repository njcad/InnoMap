import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader
from model import GCNLinkPredictor
from tqdm import tqdm

def sample_negative_edges(num_nodes, edge_index, num_samples):
    edge_set = set([tuple(e) for e in edge_index.t().tolist()])
    negative_edges = []
    while len(negative_edges) < num_samples:
        u, v = torch.randint(0, num_nodes, (2,))
        if (u.item(), v.item()) not in edge_set and (v.item(), u.item()) not in edge_set and u.item() != v.item():
            negative_edges.append((u.item(), v.item()))
    return torch.tensor(negative_edges, dtype=torch.long)

def train_with_neighborloader(model, graph, optimizer, num_epochs, batch_size=16, num_neighbors=[10,10]):
    train_loader = NeighborLoader(
        graph,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=None, 
        shuffle=True
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_auc = 0.0
        count = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch') as pbar:
            for step, batch_data in enumerate(train_loader):
                pos_edges = batch_data.edge_index
                neg_edges = sample_negative_edges(batch_data.num_nodes, pos_edges, pos_edges.size(1))
                neg_edges = neg_edges.t()

                all_edges = torch.cat([pos_edges, neg_edges], dim=1)
                labels = torch.cat([
                    torch.ones(pos_edges.size(1), device=batch_data.x.device),
                    torch.zeros(neg_edges.size(1), device=batch_data.x.device)
                ])

                optimizer.zero_grad()
                link_probs = model(batch_data.x, batch_data.edge_index, all_edges.t()).squeeze()
                loss = F.binary_cross_entropy(link_probs, labels)
                loss.backward()
                optimizer.step()

                # auc = roc_auc_score(labels.cpu().numpy(), link_probs.detach().cpu().numpy())
                # auc = roc_auc_score(labels.mps().numpy(), link_probs.detach().mps().numpy())
                auc = roc_auc_score(labels.numpy(), link_probs.detach().numpy())
                total_loss += loss.item()
                total_auc += auc
                count += 1

                pbar.update(1)
                # Update postfix only every 1000 steps to avoid too frequent updates
                if (step + 1) % 1000 == 0:
                    avg_loss = total_loss / count
                    avg_auc = total_auc / count
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", auc=f"{avg_auc:.4f}")

        # Print/checkpoint every 5 epochs
        avg_loss = total_loss / count
        avg_auc = total_auc / count
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, AUC: {avg_auc:.4f}")
        checkpoint_path = f"../checkpoints/model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    checkpoint_path = f"../checkpoints/model_epoch_{num_epochs}.pth"
    torch.save(model.state_dict(), checkpoint_path)

def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    data_path = "../data/graphs/full.pt"
    graph = torch.load(data_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('mps')
    graph = graph.to(device)
    print(f"Using device: {device}")

    model = GCNLinkPredictor(
        in_channels=graph.x.size(-1), 
        hidden_channels=64,
        out_channels=32,
    ).to(device)
    print("Model initialized")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print("Optimizer initialized")

    train_with_neighborloader(model, graph, optimizer, num_epochs=100, batch_size=16, num_neighbors=[10,10])
    print("Training completed")

if __name__ == "__main__":
    main()







### NOTE: potentially useful later???
# def predict_links(model, data, new_paper_embedding=None):
#     """
#     Predict links for a new paper or existing graph.
#     """
#     model.eval()
#     if new_paper_embedding is not None:
#         new_embedding = torch.tensor(new_paper_embedding, dtype=torch.float).unsqueeze(0)
#         data.x = torch.cat([data.x, new_embedding], dim=0)
#         new_node_idx = data.x.size(0) - 1
#         candidate_edges = torch.tensor([[new_node_idx, i] for i in range(new_node_idx)], dtype=torch.long)
#     else:
#         candidate_edges = torch.cartesian_prod(torch.arange(data.num_nodes), torch.arange(data.num_nodes))

#     with torch.no_grad():
#         link_probs = model(data.x, data.edge_index, candidate_edges)
#     return link_probs