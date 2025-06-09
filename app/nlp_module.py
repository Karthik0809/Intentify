import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import sys
import os
from functools import lru_cache
import time
import pickle
from sklearn.preprocessing import LabelEncoder
import torch.serialization

# Add parent directory to sys.path to import app_config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app_config import INTENTS

# Add LabelEncoder to safe globals for PyTorch 2.6+
torch.serialization.add_safe_globals([LabelEncoder])


class IntentClassifier(nn.Module):
    def __init__(self, num_classes=len(INTENTS), dropout_rate=0.1):
        super(IntentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        self.label_encoder = None

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use [CLS] token output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


@lru_cache(maxsize=1)
def get_tokenizer():
    """Get or create a cached tokenizer instance."""
    try:
        # Try to load from cache first
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        os.makedirs(cache_dir, exist_ok=True)
        
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            cache_dir=cache_dir,
            local_files_only=True
        )
    except Exception:
        # If not in cache, download with retries
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                tokenizer = BertTokenizer.from_pretrained(
                    "bert-base-uncased",
                    cache_dir=cache_dir
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to load tokenizer after {max_retries} attempts: {str(e)}")
                time.sleep(retry_delay * (2 ** attempt))
    
    return tokenizer


def create_label_encoder():
    """Create a new label encoder with the default intents."""
    label_encoder = LabelEncoder()
    label_encoder.fit(INTENTS)
    return label_encoder


def load_model_and_encoder(model_path="models/intent_classifier_best.pt", encoder_path="models/label_encoder.pkl"):
    """Load the model and label encoder."""
    try:
        # Load label encoder
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
        
        # Load label encoder from pickle file
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        if label_encoder is None:
            raise ValueError("Failed to load label encoder")
        
        # Load model and its state
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load the saved dictionary with weights_only=False for PyTorch 2.6+
        saved_dict = torch.load(model_path, weights_only=False)
        
        # Initialize model with correct number of classes
        model = IntentClassifier(num_classes=saved_dict['num_classes'])
        
        # Load state dict
        model.load_state_dict(saved_dict['model_state_dict'])
        
        # Set label encoder
        model.label_encoder = label_encoder
        
        # Set model to evaluation mode
        model.eval()
        
        # Verify model and encoder are properly loaded
        if model.label_encoder is None:
            raise ValueError("Label encoder not properly set in model")
        
        return model, label_encoder
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


def classify_intent(text, model, return_confidence=False):
    """Classify the intent of the given text using the trained model."""
    if model is None:
        raise ValueError("Model not loaded. Please load the model first.")
    
    if model.label_encoder is None:
        raise ValueError("Label encoder not properly initialized in model")
    
    model.eval()
    
    try:
        # Get cached tokenizer
        tokenizer = get_tokenizer()
        
        # Tokenize and prepare input
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors="pt"
        )
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(encoding['input_ids'], encoding['attention_mask'])
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get the predicted intent
        predicted_intent = model.label_encoder.inverse_transform([predicted_class])[0]
        
        if return_confidence:
            return predicted_intent, confidence
        return predicted_intent
        
    except Exception as e:
        raise Exception(f"Error during classification: {str(e)}")
