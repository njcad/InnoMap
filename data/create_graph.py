"""
This script reads a specified number of initial papers from a cleaned dataset and iteratively includes references up to 
K layers deep to create a directed graph using PyTorch Geometric. Each node represents a paper with features encoded 
from its title and abstract using a Hugging Face transformer model. Directed edges are created from referenced papers 
to referencing papers (e.g., B -> A if paper A references paper B). The processed graph is saved as 'graph_data.pt'.
"""

import pandas as pd
import torch
from torch_geometric.data import Data
# from transformers import AutoTokenizer, AutoModel # not needed w mini lm v6
import torch.nn.functional as F
from tqdm import tqdm
import ast

# # Argument dictionary to specify the number of initial papers and depth of references
# args = {
#     'num_initial_papers': 10,  # Number of initial papers to start with
#     'reference_depth': 5,        # Number of layers deep to include references
#     'save_path': './graphs/graph_data_new.pt'
# }

# # Read the cleaned data
data_clean_path = './data_clean.csv'
df = pd.read_csv(data_clean_path)
# initial_df = df.head(args['num_initial_papers'])

# # Collect initial paper IDs
# collected_ids = set(initial_df['id'].tolist())
# new_ids = collected_ids.copy()

# # Iteratively include references up to K layers deep
# for _ in range(args['reference_depth']):
#     referenced_ids = set()

#     # Find all the referenced ids and append to referenced_ids set
#     for _, row in df[df['id'].isin(new_ids)].iterrows():
#         references = ast.literal_eval(row['references']) #BLACK MAGIC LINE, I DONT KNOW HOW THIS WORKS
#         referenced_ids.update(references)  # Directly add the list to the set

#     # Find new IDs that haven't been collected yet and store in new_ids set
#     new_ids = referenced_ids - collected_ids

#     # If there are no new ids, break
#     if not new_ids:
#         break

#     # Update the collected ids with the new ids
#     collected_ids.update(new_ids)

# # Filter the dataset to only include the collected IDs
# combined_df = df[df['id'].isin(collected_ids)]
combined_df = df

# Build a mapping from paper IDs to node indices
paper_ids = combined_df['id'].tolist() # list ofpaper ids 

id_to_index = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}# map paper ids to indices in the list paper_ids

# Initialize lists to hold edge indices
edge_index = [[], []]  # [source_nodes, target_nodes]

# Build the edge index
for idx, row in combined_df.iterrows():
    curr_paper = row['id']
    source_idx = id_to_index[curr_paper]
    references = ast.literal_eval(row['references']) #BLACK MAGIC LINE, I DONT KNOW HOW THIS WORKS
    for ref_id in references:
        if ref_id in id_to_index:
            target_idx = id_to_index[ref_id]
            # Edge from referenced paper to referencing paper
            edge_index[0].append(target_idx)
            edge_index[1].append(source_idx)

# Convert edge_index to a tensor
edge_index = torch.tensor(edge_index, dtype=torch.long) #made long since colab did so too lol...

### OLD METHOD 
# NOTE: still very computationally expensive
# Load a basic Hugging Face encoder (e.g., DistilBERT)
# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# model = AutoModel.from_pretrained('distilbert-base-uncased')


### NEW (old) METHOD: nvidia NV-Embed-v2
# NOTE: too large to run locally.
# model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
# max_length = 32768


### NEW METHOD: all-mini-lm-v6
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# Prepare a list to hold node features
node_features = []

# Encode the title and abstract to create node features
print(torch.backends.mps.is_available())
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cpu') # cpu actually seems faster than gpu for this task? maybe bc of overhead of gpu? Maybe I am just not using the gpu correctly?

print(f"Using device: {device}")
model = model.to(device)  # Move model to GPU if available

batch_size = 128  # Define a batch size for encoding
texts = []

for idx, row in tqdm(combined_df.iterrows(), total=len(combined_df)):
    title = str(row['title'])
    abstract = str(row['abstract'])
    text = title + ' ' + abstract # just concat the title and abstract with a space in between for now
    texts.append(text)

    # Process in batches
    if len(texts) == batch_size:
        embeddings = model.encode(texts, batch_size=batch_size, convert_to_tensor=True, device=device)
        embeddings = F.normalize(embeddings, dim=1)
        node_features.extend(embeddings.cpu())
        texts = []

# Process any remaining texts
if texts:
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_tensor=True, device=device)
    embeddings = F.normalize(embeddings, dim=1)
    node_features.extend(embeddings.cpu())

# Stack node features into a tensor
x = torch.stack(node_features)

# Create the PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index)

# Save the data object with name indicating initial papers and depth
# save_path = f"./graphs/first_{args['num_initial_papers']}_{args['reference_depth']}.pt"
save_path = "./graphs/full.pt"
torch.save(data, save_path)

print(f"Graph data has been saved to '{save_path}'.")
