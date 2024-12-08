import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
###################################################
# Configuration
###################################################
GRAPH_TRAIN_PATH = "../data/graphs/full.pt"
GRAPH_EVAL_PATH = "../data/graphs/eval_graph.pt"
DATA_CLEAN_PATH = "../data/data_clean.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
MODEL_CHECKPOINT = "../checkpoints/linear_model_epoch_1.pth"  # save your model here
BATCH_SIZE = 64  # If needed, though we can do full-batch for simplicity

###################################################
# Model Definition
###################################################
class LinearLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, node_pairs):
        # node_pairs: [E, 2]
        src = x[node_pairs[:, 0]]
        dst = x[node_pairs[:, 1]]
        out = torch.cat([src, dst], dim=-1)
        return self.mlp(out).squeeze(-1)


###################################################
# Utility Functions
###################################################
def load_graph(path):
    print("\nLoading graph from:", path)
    graph = torch.load(path)
    return graph.to(DEVICE)

def sample_negative_edges(graph, num_samples):
    """
    Samples `num_samples` negative edges from the graph.
    """
    num_nodes = graph.num_nodes
    edge_set = set((u.item(), v.item()) for u, v in graph.edge_index.t())
    negative_edges = []
    with tqdm(total=num_samples, desc="Sampling negative edges", unit="edge") as pbar:
        while len(negative_edges) < num_samples:
            uv = torch.randint(0, num_nodes, (num_samples, 2))
            uv = uv[uv[:, 0] != uv[:, 1]]  # remove self-loops
            for u, v in uv.tolist():
                if (u, v) not in edge_set and (v, u) not in edge_set:
                    negative_edges.append((u, v))
                    pbar.update(1)
                    if len(negative_edges) >= num_samples:
                        break

    return torch.tensor(negative_edges, dtype=torch.long, device=DEVICE)

def train(model, graph, optimizer, epochs=EPOCHS, batch_size=BATCH_SIZE):
    model.train()
    x = graph.x
    pos_edges = graph.edge_index.t()  # [num_edges, 2]

    for epoch in tqdm(range(1, epochs+1), desc="Training Epochs"):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        # Sample negative edges (same number as positive edges)
        neg_edges = sample_negative_edges(graph, pos_edges.size(0))
        
        # Combine positive and negative edges and shuffle
        all_edges = torch.cat([pos_edges, neg_edges], dim=0)
        labels = torch.cat([
            torch.ones(pos_edges.size(0), device=DEVICE),
            torch.zeros(neg_edges.size(0), device=DEVICE)
        ])
        
        # Shuffle the data
        perm = torch.randperm(len(all_edges), device=DEVICE)
        all_edges = all_edges[perm]
        labels = labels[perm]

        # Calculate number of batches
        total_edges = all_edges.size(0)
        num_batches = (total_edges + batch_size - 1) // batch_size

        all_preds = []
        all_labels = []

        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch}/{epochs} Batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_edges)
            
            batch_edges = all_edges[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]

            optimizer.zero_grad()
            
            # Get predictions
            preds = model(x, batch_edges)
            
            # Compute loss
            loss = F.binary_cross_entropy(preds, batch_labels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()

            epoch_loss += loss.item()
            all_preds.append(preds.detach())
            all_labels.append(batch_labels)

        # Compute epoch statistics
        epoch_loss /= num_batches
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        auc = roc_auc_score(all_labels, all_preds)

        print(f"\nEpoch {epoch}/{epochs} Statistics:")
        print(f"Loss: {epoch_loss:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Mean prediction: {all_preds.mean():.4f}")
        print(f"Min prediction: {all_preds.min():.4f}")
        print(f"Max prediction: {all_preds.max():.4f}")
        print(f"Std prediction: {all_preds.std():.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_CHECKPOINT)
    print(f"\nModel saved to {MODEL_CHECKPOINT}")

def evaluate(model, graph, batch_size=10000):
    model.eval()
    x = graph.x
    pos_edges = graph.edge_index.t()
    neg_edges = sample_negative_edges(graph, pos_edges.size(0))
    all_edges = torch.cat([pos_edges, neg_edges], dim=0)
    labels = torch.cat([
        torch.ones(pos_edges.size(0), device=DEVICE),
        torch.zeros(neg_edges.size(0), device=DEVICE)
    ])

    total_edges = all_edges.size(0)
    num_batches = (total_edges + batch_size - 1) // batch_size

    all_preds = []

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating Batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_edges)
            batch_edges = all_edges[start_idx:end_idx]
            batch_preds = model(x, batch_edges)
            all_preds.append(batch_preds)

    all_preds = torch.cat(all_preds)
    
    # Print prediction statistics
    print("\nPrediction Statistics:")
    print(f"Mean prediction: {all_preds.mean().item():.4f}")
    print(f"Min prediction: {all_preds.min().item():.4f}")
    print(f"Max prediction: {all_preds.max().item():.4f}")
    print(f"Std prediction: {all_preds.std().item():.4f}")
    
    # Calculate optimal threshold using validation set predictions
    thresholds = torch.linspace(all_preds.min().item(), all_preds.max().item(), 100, device=DEVICE)
    best_f1 = 0
    best_threshold = thresholds[0].item()
    best_precision = 0
    best_recall = 0
    
    print("\nFinding optimal threshold...")
    for threshold in thresholds:
        pred_labels = (all_preds >= threshold).float()
        tp = ((pred_labels == 1) & (labels == 1)).sum().float()
        fp = ((pred_labels == 1) & (labels == 0)).sum().float()
        fn = ((pred_labels == 0) & (labels == 1)).sum().float()
        tn = ((pred_labels == 0) & (labels == 0)).sum().float()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold.item()
            best_precision = precision.item()
            best_recall = recall.item()

    # Use the best threshold for final predictions
    pred_labels = (all_preds >= best_threshold).cpu().long()
    all_preds = all_preds.cpu()
    labels = labels.cpu()

    auc = roc_auc_score(labels.numpy(), all_preds.numpy())
    cm = confusion_matrix(labels.numpy(), pred_labels.numpy())

    print("\nEvaluation Results:")
    print(f"AUC: {auc:.4f}")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Precision: {best_precision:.4f}")
    print(f"Best Recall: {best_recall:.4f}")
    print("Confusion Matrix:\n", cm)

    # Calculate percentages for confusion matrix
    total = cm.sum()
    print("\nConfusion Matrix Percentages:")
    print(f"True Negatives: {(cm[0,0]/total)*100:.2f}%")
    print(f"False Positives: {(cm[0,1]/total)*100:.2f}%")
    print(f"False Negatives: {(cm[1,0]/total)*100:.2f}%")
    print(f"True Positives: {(cm[1,1]/total)*100:.2f}%")

    # Save confusion matrix
    df_cm = pd.DataFrame(cm, index=["Non-edge", "Edge"], columns=["Pred Non-edge", "Pred Edge"])
    os.makedirs("./results", exist_ok=True)
    df_cm.to_csv("./results/baseline_confusion_matrix.csv")

    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("./results/baseline_confusion_matrix.png")
    plt.close()

    return auc, best_threshold, best_f1


###################################################
# Main
###################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_only', action='store_true', help='Skip training and only run evaluation')
    args = parser.parse_args()

    # Load graphs
    train_graph = load_graph(GRAPH_TRAIN_PATH)
    eval_graph = load_graph(GRAPH_EVAL_PATH)

    # Model init
    in_channels = train_graph.x.size(-1)
    hidden_channels = 64
    model = LinearLinkPredictor(in_channels, hidden_channels).to(DEVICE)

    if not args.eval_only:
        print("\n=== Training Phase ===")
        # Initialize optimizer with a smaller learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # Train for just one epoch
        train(model, train_graph, optimizer, epochs=1)
        print("\nTraining completed. Model saved.")
    else:
        print("\n=== Loading Pre-trained Model ===")
        model.load_state_dict(torch.load(MODEL_CHECKPOINT))
        model.eval()

    print("\n=== Evaluation Phase ===")
    # Evaluate on training graph
    print("\nEvaluating on training graph:")
    train_auc, train_threshold, train_f1 = evaluate(model, train_graph)

    # Evaluate on eval graph
    print("\nEvaluating on evaluation graph:")
    eval_auc, eval_threshold, eval_f1 = evaluate(model, eval_graph)
