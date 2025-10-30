"""
Tor Traffic Detection with CodeBERT + BiLSTM + Attention
---------------------------------------------------------

This script implements a deep learning pipeline to classify Tor vs. non-Tor traffic.
It combines pretrained CodeBERT embeddings with a bi-directional LSTM and an
attention mechanism for sequence modeling.

Key steps:
1. Load and preprocess traffic data from CSV files.
2. Extract handcrafted network flow features (duration, ports, throughput, timing, asymmetry, burstiness).
3. Convert numerical features into descriptive text and tokenize them with CodeBERT.
4. Train a CodeBERT + BiLSTM + Attention model with class balancing and gradient accumulation.
5. Save the best-performing model and evaluate accuracy and F1 score on a test set.

Saved model: `best_tor_codebert_model.pt`
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import warnings
from tqdm.auto import tqdm
import logging
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.WARNING)


class TorFeatureExtractor:
    @staticmethod
    def load_and_clean_data(paths):
        """Load and preprocess Tor traffic data from CSV files"""
        dfs = []
        for path in paths:
            df = pd.read_csv(path)
            # Standardize labels: nonTOR → 0, TOR → 1
            df['label'] = np.where(df['label'].str.contains('nonTOR', case=False), 0, 1)
            # Remove leading spaces from column names
            df.columns = [col.lstrip(' ') if col.startswith(' ') else col for col in df.columns]
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
        print(f"Loaded dataset: {combined.shape}")
        print(f"Label distribution:\n{combined['label'].value_counts()}")
        print(combined.head(5))
        print(combined.columns.tolist())
        return combined

    @staticmethod
    def extract_tor_features(df):
        """Extract custom Tor traffic features"""
        # Network connection features
        df['short_flow'] = (df['Flow Duration'] < 2).astype(int)
        df['high_port'] = (df['Destination Port'] > 49152).astype(int)

        # Traffic pattern features
        df['low_throughput'] = ((df['Flow Bytes/s'] < 1024) |
                                (df['Flow Packets/s'] < 10)).astype(int)

        # Timing features
        df['consistent_timing'] = ((df['Flow IAT Std'] / df['Flow IAT Mean'].clip(lower=1e-5) < 0.5) &
                                   (df['Flow IAT Mean'] > 0)).astype(int)

        # Flow asymmetry
        df['flow_asymmetry'] = (abs(df['Fwd IAT Mean'] - df['Bwd IAT Mean']) > 0.1).astype(int)

        # Burst detection
        df['bursty'] = ((df['Active Max'] / df['Active Mean'].clip(lower=1e-5) > 5) &
                        (df['Active Mean'] > 0)).astype(int)

        return df


class TorTrafficDataset(Dataset):
    """Custom dataset that converts Tor traffic features into text for CodeBERT"""

    def __init__(self, df, feature_columns, tokenizer, max_length=128, target_column='label'):
        self.df = df
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Convert numerical features into descriptive text
        feature_texts = []
        for col in self.feature_columns:
            value = row[col]
            if col == 'short_flow':
                desc = "short duration" if value else "normal duration"
            elif col == 'high_port':
                desc = "high port number" if value else "normal port number"
            elif col == 'low_throughput':
                desc = "low throughput" if value else "normal throughput"
            elif col == 'consistent_timing':
                desc = "consistent timing" if value else "variable timing"
            elif col == 'flow_asymmetry':
                desc = "asymmetric flow" if value else "symmetric flow"
            elif col == 'bursty':
                desc = "bursty traffic" if value else "steady traffic"
            else:
                desc = f"{col.replace('_', ' ')} of {value:.2f}"

            feature_texts.append(desc)

        # Combine all features into a single text
        text = "This network flow has: " + ", ".join(feature_texts) + "."

        # Tokenize with CodeBERT
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(row[self.target_column], dtype=torch.long)
        }


class CodeBertLSTM(nn.Module):
    """CodeBERT + BiLSTM + Attention classifier for Tor detection"""

    def __init__(self, config):
        super().__init__()
        self.codebert = AutoModel.from_pretrained(config.bert_path)
        for param in self.codebert.parameters():
            param.requires_grad = False

        # BiLSTM with projection
        self.lstm = nn.LSTM(
            input_size=self.codebert.config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
            proj_size=128
        )

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )

        # Final classifier
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)  # (batch_size, seq_len, 256)

        # Apply attention
        attention_weights = self.attention(lstm_out)
        attention_weights = attention_weights.transpose(1, 2)
        weighted = torch.bmm(attention_weights, lstm_out)
        features = weighted.squeeze(1)

        return self.fc(self.dropout(features))


class Config:
    """Configuration for model and training"""

    def __init__(self):
        self.max_len = 128
        self.batch_size = 32
        self.epochs = 1
        self.learning_rate = 2e-5
        self.hidden_size = 256
        self.bert_path = "../codebert/small-v2"
        self.device = torch.device('cpu')
        self.random_state = 42

        torch.set_num_threads(14)
        os.environ['OMP_NUM_THREADS'] = '14'
        os.environ['MKL_NUM_THREADS'] = '14'
        os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'


def create_data_loaders(df, feature_columns, tokenizer, config):
    """Split dataset and create DataLoaders with class balancing"""
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=config.random_state)
    train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df['label'], random_state=config.random_state)

    train_dataset = TorTrafficDataset(train_df, feature_columns, tokenizer, config.max_len)
    val_dataset = TorTrafficDataset(val_df, feature_columns, tokenizer, config.max_len)
    test_dataset = TorTrafficDataset(test_df, feature_columns, tokenizer, config.max_len)

    # Handle class imbalance
    class_counts = torch.bincount(torch.tensor(train_df['label'].values))
    weights = 1. / class_counts.float()
    samples_weights = weights[torch.tensor(train_df['label'].values)]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size * 2,
        sampler=WeightedRandomSampler(samples_weights, len(samples_weights)),
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
        multiprocessing_context='forkserver'
    )

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, config):
    """Train the model with class balancing and validation"""
    class_weights = torch.tensor([1.0, 3.0]).to(config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, foreach=True)

    accumulation_steps = 2
    best_f1 = 0.0
    grad_clip_value = 1.0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs}', unit='batch') as pbar:
            for batch_idx, batch in enumerate(pbar):
                inputs = {
                    'input_ids': batch['input_ids'].to(config.device, non_blocking=True),
                    'attention_mask': batch['attention_mask'].to(config.device, non_blocking=True)
                }
                labels = batch['labels'].to(config.device, non_blocking=True)

                with torch.cpu.amp.autocast(enabled=config.device.type == 'cpu'):
                    outputs = model(**inputs)
                    loss = criterion(outputs, labels) / accumulation_steps

                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item() * accumulation_steps
                pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

        val_metrics = evaluate_model(model, val_loader, config.device)
        print(f"Epoch {epoch + 1} | Loss: {epoch_loss/len(train_loader):.4f} | Val F1: {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'best_tor_codebert_model.pt')


def evaluate_model(model, data_loader, device):
    """Evaluate model performance on validation/test set"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }


def main():
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    # 1. Load and preprocess data
    df = TorFeatureExtractor.load_and_clean_data(['data.csv'])
    df = TorFeatureExtractor.extract_tor_features(df)

    # 2. Select features and create loaders
    feature_cols = ['short_flow', 'high_port', 'low_throughput',
                    'consistent_timing', 'flow_asymmetry', 'bursty',
                    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s']

    train_loader, val_loader, test_loader = create_data_loaders(df, feature_cols, tokenizer, config)

    # 3. Initialize and train model
    model = CodeBertLSTM(config).to(config.device)
    train_model(model, train_loader, val_loader, config)

    # 4. Evaluate on test set
    model.load_state_dict(torch.load('best_tor_codebert_model.pt'))
    test_metrics = evaluate_model(model, test_loader, config.device)

    print("\nFinal Test Metrics:")
    for k, v in test_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")


if __name__ == '__main__':
    main()
