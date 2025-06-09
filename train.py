import os
import sys
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import nltk
from nltk.corpus import wordnet
import re
from collections import Counter
import pickle
import torch.serialization

# Add LabelEncoder to safe globals for PyTorch 2.6+
torch.serialization.add_safe_globals([LabelEncoder])

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Ensure the `app/` directory is in the Python path
sys.path.insert(0, os.path.abspath('./app'))

# Import from app folder
from app_config import INTENTS, MODEL_PATH
from assistant_utils import load_snips_data, save_model, load_model
from nlp_module import IntentClassifier

def get_synonyms(word):
    """Get synonyms for a word using WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

def augment_text(text, intent=None):
    """Enhanced text augmentation with multiple techniques and intent-specific augmentation"""
    augmented = text
    
    # Intent-specific augmentation
    if intent == "AddToPlaylist":
        # Add more playlist-related variations
        if random.random() < 0.3:
            prefixes = ["add", "put", "include", "queue", "add to my"]
            suffixes = ["to my playlist", "to the playlist", "to my music", "to my collection"]
            augmented = f"{random.choice(prefixes)} {augmented} {random.choice(suffixes)}"
    
    elif intent == "SearchScreeningEvent":
        # Add more movie-related variations
        if random.random() < 0.3:
            prefixes = ["find", "search for", "look for", "check", "show me"]
            suffixes = ["near me", "in my area", "around here", "in this city", "in the vicinity"]
            augmented = f"{random.choice(prefixes)} {augmented} {random.choice(suffixes)}"
    
    # Randomly apply different augmentation techniques
    if random.random() < 0.3:
        # Synonym replacement
        words = augmented.split()
        for i, word in enumerate(words):
            if random.random() < 0.3:
                synonyms = get_synonyms(word)
                if synonyms:
                    words[i] = random.choice(synonyms)
        augmented = ' '.join(words)
    
    if random.random() < 0.3:
        # Random case changes
        words = augmented.split()
        augmented = ' '.join(word.upper() if random.random() < 0.3 else word for word in words)
    
    if random.random() < 0.3:
        # Add/remove spaces
        augmented = re.sub(r'\s+', ' ', augmented)
        if random.random() < 0.5:
            augmented = ' '.join(augmented.split())
        else:
            augmented = augmented.replace(' ', '  ')
    
    if random.random() < 0.3:
        # Add punctuation variations
        if not augmented.endswith(('.', '!', '?')):
            augmented += random.choice(['.', '!', '?'])
    
    if random.random() < 0.3:
        # Add/remove common words
        common_words = ['please', 'kindly', 'could you', 'would you', 'i want to', 'i need to']
        if random.random() < 0.5:
            # Add common word
            augmented = f"{random.choice(common_words)} {augmented}"
        else:
            # Remove common word if present
            for word in common_words:
                augmented = augmented.replace(word, '').strip()
    
    return augmented

class SnipsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_encoder, augment=False, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.augment = augment
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.augment:
            text = augment_text(text, label)
            
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.label_encoder.transform([label])[0], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

def train_fold(model, train_loader, val_loader, device, fold, n_folds):
    """Train model for a single fold"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Use weighted loss for imbalanced classes
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['labels'].tolist())
    class_counts = Counter(all_labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    weights = torch.tensor([class_weights[cls] for cls in range(len(class_counts))]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    
    best_val_loss = float("inf")
    patience, patience_counter = 10, 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\n📊 Training Fold {fold + 1}/{n_folds}")
    
    for epoch in range(30):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            
            l1_lambda = 0.001
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss)
        
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"📉 Learning rate decreased from {current_lr:.2e} to {new_lr:.2e}")
            current_lr = new_lr

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} | Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save model state dict and label encoder together
            model_save_dict = {
                'model_state_dict': model.state_dict(),
                'num_classes': model.classifier[-1].out_features
            }
            torch.save(model_save_dict, f"models/intent_classifier_fold_{fold+1}.pt")
            print(f"💾 Saved best model checkpoint for fold {fold+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping triggered for fold {fold+1}")
                break
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies

def train():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Load data
    texts, labels = load_snips_data("data/snips_augmented.json")
    if not texts or not labels:
        print("No data found. Please check the data file.")
        return
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    # Save label encoder
    os.makedirs("models", exist_ok=True)
    with open("models/label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    print("💾 Saved label encoder to models/label_encoder.pkl")

    # Initialize K-Fold cross validation
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
        # Split data for this fold
        X_train, X_val = [texts[i] for i in train_idx], [texts[i] for i in val_idx]
        y_train, y_val = [labels[i] for i in train_idx], [labels[i] for i in val_idx]
        
        # Create datasets with class weights
        train_dataset = SnipsDataset(X_train, y_train, tokenizer, label_encoder, augment=True)
        val_dataset = SnipsDataset(X_val, y_val, tokenizer, label_encoder, augment=False)
        
        # Calculate class weights for sampling
        class_counts = Counter(y_train)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Initialize model for this fold
        model = IntentClassifier(num_classes=len(label_encoder.classes_), dropout_rate=0.1).to(device)
        
        # Train the model
        model, train_losses, val_losses, train_accs, val_accs = train_fold(
            model, train_loader, val_loader, device, fold, n_folds
        )
        
        fold_results.append({
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        })
    
    # Plot results for all folds
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    for i, result in enumerate(fold_results):
        plt.plot(result['train_losses'], label=f'Train Fold {i+1}', alpha=0.3)
        plt.plot(result['val_losses'], label=f'Val Fold {i+1}', alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Across Folds')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(2, 1, 2)
    for i, result in enumerate(fold_results):
        plt.plot(result['train_accs'], label=f'Train Fold {i+1}', alpha=0.3)
        plt.plot(result['val_accs'], label=f'Val Fold {i+1}', alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Across Folds')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cross_validation_results.png')
    plt.show()
    
    # Save the best model from all folds
    best_fold = max(range(n_folds), key=lambda i: max(fold_results[i]['val_accs']))
    best_model = fold_results[best_fold]['model']
    
    # Save model state dict and label encoder together
    model_save_dict = {
        'model_state_dict': best_model.state_dict(),
        'num_classes': best_model.classifier[-1].out_features
    }
    torch.save(model_save_dict, "models/intent_classifier_best.pt")
    print(f"\n✅ Training completed. Best model from fold {best_fold + 1} saved to models/intent_classifier_best.pt")
    
    # Final evaluation on the best model
    best_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = best_model(input_ids, attention_mask)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))

    print("🧩 Confusion Matrix:")
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=label_encoder.classes_)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    train()
