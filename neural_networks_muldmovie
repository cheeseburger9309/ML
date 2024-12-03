import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }

class FeedForwardClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self.ff_layers = nn.Sequential(
            nn.Linear(embedding_dim * 128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        embedded = self.layer_norm(embedded)
        embedded = embedded * attention_mask.unsqueeze(-1)
        embedded = embedded.view(embedded.shape[0], -1)
        return self.ff_layers(embedded)

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, num_classes, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, n_filters, fs) for fs in filter_sizes
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(len(filter_sizes) * n_filters, n_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_filters, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        embedded = self.layer_norm(embedded)
        embedded = embedded.permute(0, 2, 1)
        
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = torch.cat(pooled, dim=1)
        return self.fc(cat)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        embedded = self.layer_norm(embedded)
        
        packed_output, (hidden, _) = self.lstm(embedded)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden)

class ModelTrainer:
    def __init__(self, model, device, model_name):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def train(self, train_loader, valid_loader, num_epochs, learning_rate=2e-5):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}'):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            val_loss, val_accuracy = self.evaluate(valid_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            scheduler.step(val_loss)
            
            logger.info(f'Epoch {epoch+1}:')
            logger.info(f'Average training loss: {avg_train_loss:.4f}')
            logger.info(f'Validation loss: {val_loss:.4f}')
            logger.info(f'Validation accuracy: {val_accuracy:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f'best_{self.model_name}_model.pt')
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                _, predictions = torch.max(outputs, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.shape[0]
                total_loss += loss.item()
        
        return total_loss / len(data_loader), correct_predictions / total_predictions
    
    def predict(self, data_loader):
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        return predictions, true_labels

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.close()

def plot_confusion_matrix(true_labels, predictions, label_names, model_name):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

def generate_classification_report(true_labels, predictions, label_names, model_name):
    report = classification_report(true_labels, predictions, 
                                 target_names=label_names, 
                                 output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f'{model_name}_classification_report.csv')
    return df_report

def prepare_data(dataset):
    texts = []
    labels = []
    
    for item in dataset:
        text = item['input']
        label = item['output'][0] if item['output'] else None
        
        if text and label:
            texts.append(text)
            labels.append(label)
    
    return texts, labels

def initialize_tokenizer():
    try:
        # Try using BERT tokenizer
        logger.info("Initializing BERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception as e:
        logger.error(f"Error initializing BERT tokenizer: {str(e)}")
        raise
    return tokenizer

def get_model_configs(tokenizer, num_classes):
    return {
        'feedforward': {
            'vocab_size': tokenizer.vocab_size,
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_classes': num_classes,
            'dropout': 0.5
        },
        'cnn': {
            'vocab_size': tokenizer.vocab_size,
            'embedding_dim': 256,
            'n_filters': 100,
            'filter_sizes': [3, 4, 5],
            'num_classes': num_classes,
            'dropout': 0.5
        },
        'lstm': {
            'vocab_size': tokenizer.vocab_size,
            'embedding_dim': 256,
            'hidden_dim': 256,
            'num_classes': num_classes,
            'num_layers': 2,
            'dropout': 0.5
        }
    }

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading MULD dataset...")
    dataset = load_dataset("ghomasHudson/muld", name="Character Archetype Classification")
    
    # Prepare data
    train_texts, train_labels = prepare_data(dataset['train'])
    val_texts, val_labels = prepare_data(dataset['validation'])
    test_texts, test_labels = prepare_data(dataset['test'])
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer()
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    val_labels_encoded = label_encoder.transform(val_labels)
    test_labels_encoded = label_encoder.transform(test_labels)
    
    # Get model configurations
    model_configs = get_model_configs(tokenizer, len(label_encoder.classes_))
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels_encoded, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels_encoded, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels_encoded, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Train and evaluate each model
    results = {}
    for model_name, model_params in model_configs.items():
        logger.info(f"\nTraining {model_name} model...")
        
        # Initialize model
        if model_name == 'feedforward':
            model = FeedForwardClassifier(**model_params).to(device)
        elif model_name == 'cnn':
            model = CNNClassifier(**model_params).to(device)
        elif model_name == 'lstm':
            model = LSTMClassifier(**model_params).to(device)
        
        # Train model
        trainer = ModelTrainer(model, device, model_name)
        trainer.train(train_loader, val_loader, num_epochs=10)
        
        # Evaluate on test set
        predictions, true_labels = trainer.predict(test_loader)
        
        # Generate and save results
        plot_training_history(trainer.history, model_name)
        plot_confusion_matrix(true_labels, predictions, 
                            label_encoder.classes_, model_name)
        report = generate_classification_report(true_labels, predictions,
                                             label_encoder.classes_, model_name)
        results[model_name] = {'history': trainer.history, 'report': report}
    
    # Print final results
    for model_name, result in results.items():
        print(f"\nResults for {model_name}:")
        print("Test Set Performance:")
        print(result['report'])

if __name__ == "__main__":
    main()

