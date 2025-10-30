import time
import pandas as pd
import torch
from huggingface_hub.utils.tqdm import progress_bar_states
from torch import nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import base64
import re
import os
import warnings
from tqdm.auto import tqdm
import logging

# 1. Completely suppress all TensorFlow and CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # show errors only
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # explicitly select GPU
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings("ignore", category=UserWarning)


# 2. Configure more aggressive log filtering
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)


# 1. Data loading and preprocessing enhancements
class WebShellPreprocessor:
    @staticmethod
    def load_and_clean_data(paths):
        """Load and merge multiple datasets"""
        dfs = []
        for path in paths:
            df = pd.read_csv(path, header=None, names=['content', 'label'])
            df['label'] = df['label'].fillna(0).astype(int)
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        print(f"Dataset size: {combined.shape}")
        print(f"Label distribution:\n{combined['label'].value_counts()}")
        return combined

    @staticmethod
    def clean_base64(content):
        """Enhanced Base64 cleaning"""
        try:
            # Remove non-Base64 characters and pad to proper length
            content = re.sub(r'[^A-Za-z0-9+/=]', '', content)
            padding = len(content) % 4
            if padding:
                content += '=' * (4 - padding)
            return content
        except:
            return content

    @staticmethod
    def extract_webshell_features(decoded):
        """Extract key WebShell features"""
        patterns = [
            r'(eval|system|exec|shell_exec|passthru|popen|proc_open)\s*\([^)]*\)',  # dangerous PHP functions
            r'<%(.*?)%>',  # JSP tags
            r'Runtime\.getRuntime\(\)\.exec\([^)]*\)',  # Java execution
            r'(base64_decode|gzinflate|str_rot13)\([^)]*\)',  # common decoding functions
            r'(cmd\.exe|bash|powershell|wscript)',  # suspicious commands
            r'(union select|select @@version|drop table)',  # SQL injection patterns
            r'(document\.write|eval\(|fromCharCode)',  # XSS features
            r'(/bin/sh|/bin/bash)',  # shell indicators
            r'(phpspy|c99|r57|b374k)',  # common WebShell keywords
            r'(<\?php|\?>|<\?=)',  # PHP tags
            r'(\$_(GET|POST|REQUEST|COOKIE)\[)',  # superglobal usage
            r'(file_put_contents|fwrite|mkdir)\s*\([^)]*\)'  # file operations
        ]
        features = []
        for pattern in patterns:
            matches = re.findall(pattern, decoded, re.IGNORECASE)
            # flatten tuple matches when necessary
            features.extend([m[0] if isinstance(m, tuple) else m for m in matches])
        # return deduplicated matched features joined, otherwise a short uppercase excerpt of decoded text
        return ' '.join(set(features)) if features else decoded.upper()[:500]


# 2. Configuration class (unchanged semantics)
class Config:
    def __init__(self):
        self.max_len = 256
        self.batch_size = 32
        self.epochs = 3  # increase training epochs
        self.learning_rate = 2e-5
        self.hidden_size = 768  # match CodeBERT hidden size
        self.bert_path = "../codebert/small-v2"
        self.device = torch.device
        self.random_state = 42


# 3. Model architecture improvements
class CodeBertLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.codebert = AutoModel.from_pretrained(config.bert_path)
        for param in self.codebert.parameters():  # selective freezing
            param.requires_grad = False

        # Enhanced LSTM structure
        self.lstm = nn.LSTM(
            input_size=self.codebert.config.hidden_size,
            hidden_size=config.hidden_size,  # 768
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 256),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )

        # Enhanced classifier
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 256),  # *2 for bidirectional
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        # BERT layer
        outputs = self.codebert(input_ids, attention_mask=attention_mask)

        # LSTM layer
        lstm_out, _ = self.lstm(outputs.last_hidden_state)  # [batch, seq_len, hidden_size*2]

        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attention_weights = attention_weights.transpose(1, 2)  # [batch, 1, seq_len]
        weighted = torch.bmm(attention_weights, lstm_out)  # [batch, 1, hidden_size*2]
        features = weighted.squeeze(1)  # [batch, hidden_size*2]

        return self.fc(self.dropout(features))


class WebShellDetector:
    def __init__(self, config):
        """
        Initialize detector
        :param config: Config object containing model path, device, etc.
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.bert_path,
            local_files_only=True
        )
        self.model = CodeBertLSTM(config).to(config.device)

    def load_model(self, model_path):
        """
        Load pretrained model weights
        :param model_path: path to model weights (.pt or .bin)
        """
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.config.device)
            )
            self.model.eval()
            print(f"Successfully loaded model weights: {model_path}")
        except Exception as e:
            raise ValueError(f"Model loading failed: {str(e)}")

    def predict(self, base64_content):
        """
        Predict a single sample
        :param base64_content: Base64 encoded content to inspect
        :return: dict containing prediction result
        """
        try:
            start_time = time.time()

            # 1. Robust decoding pipeline
            cleaned = WebShellPreprocessor.clean_base64(base64_content)
            decoded = base64.b64decode(cleaned).decode('utf-8', errors='ignore')

            # 2. Feature extraction using the preprocessor
            features = WebShellPreprocessor.extract_webshell_features(decoded)

            # 3. Tokenize (with truncation and padding)
            encoding = self.tokenizer(
                features,
                max_length=self.config.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )

            # 4. Prediction
            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoding['input_ids'].to(self.config.device),
                    attention_mask=encoding['attention_mask'].to(self.config.device)
                )
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs).item()

            return {
                'prediction': pred,
                'confidence': probs[0][pred].item(),
                'features': features[:200] + '...' if len(features) > 200 else features,
                'time_ms': f"{(time.time() - start_time) * 1000:.2f}"
            }

        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'raw_content': base64_content[:100] + '...'  # return part of raw content for debugging
            }


# 4. Data loader optimizations
def create_data_loaders(opcode_sequences, labels, tokenizer, config):
    # stratified splits
    train_val, test = train_test_split(
        list(zip(opcode_sequences, labels)),
        test_size=0.2,
        stratify=labels,
        random_state=config.random_state
    )
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        stratify=[y for _, y in train_val],
        random_state=config.random_state
    )

    # dynamic class weights
    class_counts = torch.bincount(torch.tensor([y for _, y in train]))
    weights = 1. / class_counts.float()
    samples_weights = weights[torch.tensor([y for _, y in train])]

    # dataset class
    class WebShellDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            seq, label = self.data[idx]
            encoding = tokenizer(
                seq,
                max_length=config.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(label)
            }

    # create loaders
    train_loader = DataLoader(
        WebShellDataset(train),
        batch_size=config.batch_size,
        sampler=WeightedRandomSampler(samples_weights, len(samples_weights)),
        num_workers=4
    )
    val_loader = DataLoader(WebShellDataset(val), batch_size=config.batch_size)
    test_loader = DataLoader(WebShellDataset(test), batch_size=config.batch_size)

    return train_loader, val_loader, test_loader


# 5. Training flow improvements
def train_model(model, train_loader, val_loader, optimizer, criterion, config):
    best_f1 = 0.0
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        # use a more robust progress bar
        with tqdm(train_loader,
                  desc=f'Epoch {epoch + 1}/{config.epochs}',
                  unit='batch',
                  bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                  dynamic_ncols=True) as pbar:

            for batch in pbar:
                try:
                    optimizer.zero_grad()
                    inputs = {
                        'input_ids': batch['input_ids'].to(config.device),
                        'attention_mask': batch['attention_mask'].to(config.device)
                    }
                    labels = batch['label'].to(config.device)

                    outputs = model(**inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                except Exception as e:
                    pbar.write(f"Error in batch: {str(e)}")
                    continue

        # validation step
        val_metrics = evaluate_model(model, val_loader, config)
        print(f"\nEpoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f} | "
              f"Val F1: {val_metrics[1]:.4f}")

        if val_metrics[1] > best_f1:
            best_f1 = val_metrics[1]
            torch.save(model.state_dict(), 'best_codebert_model.pt')


def evaluate_model(model, data_loader, config):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        with tqdm(data_loader,
                  desc='Evaluating',
                  unit='batch',
                  leave=False,
                  dynamic_ncols=True) as pbar:
            for batch in pbar:
                inputs = {
                    'input_ids': batch['input_ids'].to(config.device),
                    'attention_mask': batch['attention_mask'].to(config.device)
                }
                labels = batch['label'].to(config.device)

                outputs = model(**inputs)
                _, preds = torch.max(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    # compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return accuracy, f1, recall, precision


# main entrypoint
def main():
    # initialize
    config = Config()
    preprocessor = WebShellPreprocessor()

    # data loading and preprocessing
    df = preprocessor.load_and_clean_data([
        'webshell.csv',
        'webshell_data.csv'
    ])

    opcode_sequences, labels = [], []
    for _, row in df.iterrows():
        try:
            cleaned = preprocessor.clean_base64(row['content'])
            decoded = base64.b64decode(cleaned).decode('utf-8', errors='ignore')
            features = preprocessor.extract_webshell_features(decoded)
            if features:
                opcode_sequences.append(features)
                labels.append(row['label'])
        except:
            continue

    # training pipeline
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    train_loader, val_loader, test_loader = create_data_loaders(
        opcode_sequences, labels, tokenizer, config
    )

    model = CodeBertLSTM(config).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(config.device))

    train_model(model, train_loader, val_loader, optimizer, criterion, config)

    # test evaluation
    model.load_state_dict(torch.load('best_codebert_model.pt'))
    test_metrics = evaluate_model(model, test_loader, config)
    print(f"\nTest Metrics - Acc: {test_metrics[0]:.4f} | F1: {test_metrics[1]:.4f} | "
          f"Recall: {test_metrics[2]:.4f} | Precision: {test_metrics[3]:.4f}")

    # deploy detector
    detector = WebShellDetector(config)
    detector.load_model('best_codebert_model.pt')

    test_samples = [
        "PD9waHAgZXZhbCgkX1BPU1RbJ2NtZCddKTs=",  # PHP shell
        "U0hFTEwgY21kLmV4ZQ==",  # Suspicious
        "aW5kZXguaHRtbA=="  # Normal
    ]

    print("\nSample Predictions:")
    for i, sample in enumerate(test_samples):
        result = detector.predict(sample)
        print(f"Sample {i + 1}: {'Malicious' if result.get('prediction') else 'Normal'} "
              f"(Confidence: {result.get('confidence', 0):.1%})")


if __name__ == '__main__':
    main()
