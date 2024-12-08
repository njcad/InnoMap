import gzip
import csv
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Paths
data_path = './dataset/ogbn_arxiv/processed/geometric_data_processed.pt'
nodeid2paperid_path = './dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz'
titleabs_path = './dataset/ogbn_arxiv/processed/titleabs.tsv'

# Load the original Data object
original_data = torch.load(data_path)[0]

# Load nodeidx2paperid
node_idx_to_paper_id = []
with gzip.open(nodeid2paperid_path, 'rt') as f:
    reader = csv.reader(f)
    next(reader)  # skip header if it exists
    for row in reader:
        # Format: node_idx, paper_id
        node_idx = int(row[0])
        paper_id = int(row[1])
        node_idx_to_paper_id.append((node_idx, paper_id))

# Ensure sorted by node_idx to match node ordering if needed
node_idx_to_paper_id.sort(key=lambda x: x[0])
paper_ids = [p for _, p in node_idx_to_paper_id]

# Create a mapping from paper_id to (title, abstract)
paperid_to_text = {}
with open(titleabs_path, 'r', encoding='utf-8') as f:
    # Skip the header if there is one
    header = next(f).strip().split('\t')
    # Now iterate over actual data lines
    for line in f:
        parts = line.strip().split('\t')
        # Process parts as needed
        if len(parts) < 3:
            continue
        pid = int(parts[0])
        title = parts[1]
        abstract = parts[2]
        text = title + " " + abstract
        paperid_to_text[pid] = text

# Prepare the texts in node order
texts = [paperid_to_text[pid] for pid in paper_ids]

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)

# Encode in batches
batch_size = 128
all_embeddings = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    embeddings = model.encode(batch, batch_size=batch_size, convert_to_tensor=True, device=device)
    embeddings = F.normalize(embeddings, dim=1)
    all_embeddings.append(embeddings.cpu())

# Stack into single tensor
new_x = torch.cat(all_embeddings, dim=0)

# Create a new Data object with the same edges but updated x
new_data = Data(
    x=new_x,
    edge_index=original_data.edge_index,
    y=original_data.y if original_data.y is not None else None
)

# Save the updated graph
save_path = './graphs/eval_graph.pt'
torch.save(new_data, save_path)
print(f"New graph data with embeddings saved to '{save_path}'")
