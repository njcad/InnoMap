# src/train.py

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from dataset import get_dataloader
from model import GCNEncoder, EdgePredictor, GCNLinkPredictor

def train_model(config):
    # Configuration parameters
    data_path = config['data_path']
    batch_size = config['batch_size']
    device = torch.device(config['device'])
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    encoder_hidden_channels = config['encoder_hidden_channels']
    encoder_out_channels = config['encoder_out_channels']
    predictor_hidden_channels = config['predictor_hidden_channels']
    
    # Initialize dataloader
    train_loader = get_dataloader(
        data_path=data_path,
        batch_size=batch_size,
        num_negative_samples=config.get('num_negative_samples', 1),
        mode='train'
    )
    data = train_loader.dataset.data.to(device)
    
    # Initialize model
    encoder = GCNEncoder(
        in_channels=data.x.size(-1),
        hidden_channels=encoder_hidden_channels,
        out_channels=encoder_out_channels
    )
    predictor = EdgePredictor(
        in_channels=encoder_out_channels,
        hidden_channels=predictor_hidden_channels
    )
    model = GCNLinkPredictor(encoder, predictor).to(device)
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for src_features, x_news, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'):
            src_features = src_features.to(device)
            x_news = x_news.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model.predictor(src_features, x_news)
            # Compute loss
            loss = F.binary_cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
        
        # Optionally, evaluate on validation set
        # if epoch % config.get('eval_interval', 5) == 0:
        #     val_auc = evaluate_model(model, val_loader, device)
        #     print(f'Validation AUC: {val_auc:.4f}')
    
    # Save the trained model
    model_save_path = config.get('model_save_path', './Models/gcn_link_predictor.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
    
    return model

def evaluate_model(model, dataloader, device):
    from sklearn.metrics import roc_auc_score
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for src_features, x_news, batch_labels in dataloader:
            src_features = src_features.to(device)
            x_news = x_news.to(device)
            outputs = model.predictor(src_features, x_news)
            preds.append(outputs.cpu())
            labels.append(batch_labels)
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    auc = roc_auc_score(labels.numpy(), preds.numpy())
    return auc

if __name__ == '__main__':
    # Load configuration parameters
    import yaml
    with open('./configs/defaults.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # Start training
    trained_model = train_model(config)
