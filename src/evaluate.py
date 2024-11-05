# src/evaluate.py

import torch
from dataset import get_dataloader
from model import GCNEncoder, EdgePredictor, GCNLinkPredictor

def evaluate_test_set(config):
    # Configuration parameters
    data_path = config['data_path']
    batch_size = config['batch_size']
    device = torch.device(config['device'])
    encoder_hidden_channels = config['encoder_hidden_channels']
    encoder_out_channels = config['encoder_out_channels']
    predictor_hidden_channels = config['predictor_hidden_channels']
    model_save_path = config.get('model_save_path', './Models/gcn_link_predictor.pth')
    
    # Initialize dataloader
    test_loader = get_dataloader(
        data_path=data_path,
        batch_size=batch_size,
        num_negative_samples=config.get('num_negative_samples', 1),
        mode='test'
    )
    data = test_loader.dataset.data.to(device)
    
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
    
    # Load trained model weights
    model.load_state_dict(torch.load(model_save_path))
    
    # Evaluate on test set
    test_auc = evaluate_model(model, test_loader, device)
    print(f'Test AUC: {test_auc:.4f}')

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
    # Evaluate on test set
    evaluate_test_set(config)
