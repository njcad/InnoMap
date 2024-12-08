import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import GCNLinkPredictor
from examples import startup_abstracts
from tqdm import tqdm


# Ensure results directory exists
os.makedirs("./results", exist_ok=True)

# ---- CONFIG ----
MODEL_CHECKPOINT = "../new_checkpoints/model_epoch_1.pth"  # adjust as needed
GRAPH_TRAIN_PATH = "../data/graphs/full.pt"
GRAPH_EVAL_PATH = "../data/graphs/eval_graph.pt"
DEVICE = 'cpu' #torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
DATA_CLEAN_PATH = "../data/data_clean.csv"  # Path to the CSV used to create the train graph
# NUM_NEG_SAMPLES = 100  # Adjust as needed for performance


def load_graph(path):
    print("\nLoading graph from:", path)
    return torch.load(path)

def load_model(model_path, in_channels, hidden_channels=64, out_channels=32):
    print("\nLoading model from:", model_path)
    model = GCNLinkPredictor(in_channels, hidden_channels, out_channels).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def sample_subset_negative_edges(graph):
    """
    Samples # of negative edges equal to the number of positive edges in the graph.

    Parameters:
    - graph: PyTorch Geometric Data object.
    - num_neg_samples: Number of negative edges to sample.

    Returns:
    - negative_edges: Tensor of sampled negative edges.
    """
    num_nodes = graph.num_nodes
    edge_set = set((u.item(), v.item()) for u, v in graph.edge_index.t())

    size = graph.edge_index.size(1)
    negative_edges = []
    with tqdm(total=size, desc="Sampling negative edges", unit="edge") as pbar:
        while len(negative_edges) < size:
            uv = torch.randint(0, num_nodes, (size, 2))
            uv = uv[uv[:, 0] != uv[:, 1]]  # Remove self-loops
            for u, v in uv.tolist():
                if (u, v) not in edge_set and (v, u) not in edge_set:
                    negative_edges.append((u, v))
                    pbar.update(1)
                    if len(negative_edges) >= size:
                        break

    return torch.tensor(negative_edges, dtype=torch.long)



def get_sampled_predictions(model, graph, batch_size=10000):
    print("\nGetting sampled predictions")

    # Positive edges (existing edges in the graph)
    pos_edges = graph.edge_index.t()  # shape [num_edges, 2]
    print(f"Positive edges: {pos_edges.shape}")
    # Sample a set of negative edges
    neg_edges = sample_subset_negative_edges(graph).to(DEVICE)

    # Combine positive and negative edges
    all_edges = torch.cat([pos_edges, neg_edges], dim=0).to(DEVICE)
    labels = torch.cat([
        torch.ones(pos_edges.size(0), dtype=torch.float, device=DEVICE),
        torch.zeros(neg_edges.size(0), dtype=torch.float, device=DEVICE)
    ])
    print("Making predictions for part 1 (labels were created)")

    # Make predictions in batches
    link_probs = []
    with torch.no_grad():
        with tqdm(total=all_edges.size(0), desc="Predicting link probabilities", unit="batch") as pbar:
            for i in range(0, all_edges.size(0), batch_size):
                batch_edges = all_edges[i:i+batch_size]
                batch_probs = model(graph.x, graph.edge_index, batch_edges)
                link_probs.append(batch_probs.cpu())
                pbar.update(batch_size)

    link_probs = torch.cat(link_probs)
    return all_edges.cpu(), link_probs, labels.cpu()

def save_confusion_matrix(y_true, y_pred, filepath_csv, filepath_png):
    print("\nSaving confusion matrix to:", filepath_csv)
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=["Non-edge", "Edge"], columns=["Pred Non-edge", "Pred Edge"])
    df_cm.to_csv(filepath_csv)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(filepath_png)
    plt.close()

def get_node_embeddings(model, graph):
    print("\nGetting node embeddings")
    with torch.no_grad():
        z = model.get_embeddings(graph.x, graph.edge_index)
    return z

def embed_new_texts(texts, device=DEVICE):
    print("\nEmbedding new texts")
    emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
    embeddings = emb_model.encode(texts, convert_to_tensor=True, device=device)
    embeddings = F.normalize(embeddings, dim=1)
    return embeddings.cpu()

def predict_edges_for_new_embeddings(model, graph_embeddings, new_embeddings, top_k=10):
    print("\nPredicting edges for new embeddings")
    """
    For each new embedding (treated like a target node),
    predict edges from all existing nodes to it, rank by probability.
    Return top_k predicted edges.
    """
    results = []
    num_nodes = graph_embeddings.size(0)

    with torch.no_grad():
        with tqdm(total=len(new_embeddings), desc="Predicting edges for new embeddings", unit="embedding") as pbar:
            for idx, emb in enumerate(new_embeddings):
                emb = emb.unsqueeze(0).to(DEVICE)  
                src_emb = graph_embeddings.to(DEVICE)
                target_emb = emb.repeat(num_nodes, 1)  
                link_probs = model.predict_links(src_emb, target_emb).squeeze()

                probs = link_probs.cpu().numpy()
                sorted_indices = np.argsort(-probs)  # descending order

                top_nodes = sorted_indices[:top_k]
                top_probs = probs[top_nodes]
                res = list(zip(top_nodes.tolist(), top_probs.tolist()))
                results.append((idx, res))
                pbar.update(1)
    return results

def main():
    # Load eval graph
    eval_graph = load_graph(GRAPH_EVAL_PATH).to(DEVICE)
    # Load model
    model = load_model(MODEL_CHECKPOINT, in_channels=eval_graph.x.size(-1))

    # Part 1: Full graph evaluation
    all_edges, link_probs, labels = get_sampled_predictions(model, eval_graph)
    preds = (link_probs >= 0.5).long()
    save_confusion_matrix(labels.numpy(), preds.numpy(), "./results/confusion_matrix.csv", "./results/confusion_matrix.png")
    print("Confusion matrix saved at ./results/confusion_matrix.csv and ./results/confusion_matrix.png")

    # Part 2: Predict edges for startup abstracts on the training graph
    graph_train = load_graph(GRAPH_TRAIN_PATH).to(DEVICE)
    z = get_node_embeddings(model, graph_train)

    # Load the dataframe used to build the train graph so we can get titles/abstracts by node idx
    combined_df = pd.read_csv(DATA_CLEAN_PATH)

    new_embs = embed_new_texts(startup_abstracts)
    results = predict_edges_for_new_embeddings(model, z, new_embs, top_k=10)

    with open("./results/new_text_predictions.txt", "w") as f:
        for i, node_probs in results:
            f.write("=====================================================================\n")
            f.write(f"Startup Abstract (Index {i}):\n{startup_abstracts[i]}\n\n")
            f.write("Top Predicted Nodes and Probabilities:\n")
            for node_idx, prob in node_probs:
                title = combined_df.iloc[node_idx]['title']
                abstract = combined_df.iloc[node_idx]['abstract']
                f.write(f"  Probability: {prob:.4f}\n")
                f.write(f"  Title: {title}\n")
                f.write(f"  Abstract: {abstract}\n\n")

    print("Startup edge predictions with titles and abstracts saved at ./results/new_text_predictions.txt")



if __name__ == "__main__":
    main()
