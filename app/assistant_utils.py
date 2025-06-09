import json
import torch

def load_snips_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    texts, labels = [], []
    for item in data['data']:
        texts.append(item['text'])
        labels.append(item['intent'])
    return texts, labels

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, path):
    # Load the saved dictionary
    saved_dict = torch.load(path)
    
    # Initialize model with correct number of classes
    model = model_class(num_classes=saved_dict['num_classes'])
    
    # Load state dict
    model.load_state_dict(saved_dict['model_state_dict'])
    model.eval()
    return model
