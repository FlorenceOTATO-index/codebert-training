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

# 1. 彻底抑制所有TensorFlow和CUDA的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示错误
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 明确指定GPU
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings("ignore", category=UserWarning)


# 2. 配置更彻底的日志过滤
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)


# 1. 数据加载与预处理强化
class WebShellPreprocessor:
    @staticmethod
    def load_and_clean_data(paths):
        """加载并合并多个数据集"""
        dfs = []
        for path in paths:
            df = pd.read_csv(path, header=None, names=['content', 'label'])
            df['label'] = df['label'].fillna(0).astype(int)
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        print(f"数据集大小: {combined.shape}")
        print(f"标签分布:\n{combined['label'].value_counts()}")
        return combined

    @staticmethod
    def clean_base64(content):
        """强化Base64清洗"""
        try:
            # 移除非Base64字符并补全长度
            content = re.sub(r'[^A-Za-z0-9+/=]', '', content)
            padding = len(content) % 4
            if padding: content += '=' * (4 - padding)
            return content
        except:
            return content

    @staticmethod
    def extract_webshell_features(decoded):
        """提取关键WebShell特征"""
        patterns = [
            r'(eval|system|exec|shell_exec|passthru|popen|proc_open)\s*\([^)]*\)',  # PHP危险函数
            r'<%(.*?)%>',  # JSP标签
            r'Runtime\.getRuntime\(\)\.exec\([^)]*\)',  # Java执行
            r'(base64_decode|gzinflate|str_rot13)\([^)]*\)',  # 常见编码函数
            r'(cmd\.exe|bash|powershell|wscript)',  # 可疑命令
            r'(union select|select @@version|drop table)',  # SQL注入特征
            r'(document\.write|eval\(|fromCharCode)',  # XSS特征
            r'(/bin/sh|/bin/bash)',  # Shell特征
            r'(phpspy|c99|r57|b374k)',  # 常见WebShell关键字
            r'(<\?php|\?>|<\?=)',  # PHP标记
            r'(\$_(GET|POST|REQUEST|COOKIE)\[)',  # 超全局变量
            r'(file_put_contents|fwrite|mkdir)\s*\([^)]*\)'  # 文件操作
        ]
        features = []
        for pattern in patterns:
            matches = re.findall(pattern, decoded, re.IGNORECASE)
            features.extend([m[0] if isinstance(m, tuple) else m for m in matches])
        return ' '.join(set(features)) if features else decoded.upper()[:500]  # 去重后合并


# 2. 配置类（保持不变）
class Config:
    def __init__(self):
        self.max_len = 256
        self.batch_size = 32
        self.epochs = 3  # 增加训练轮次
        self.learning_rate = 2e-5
        self.hidden_size = 768  # 与CodeBERT隐藏层一致
        self.bert_path = "../codebert/small-v2"
        self.device = torch.device
        self.random_state = 42



# 3. 模型架构优化
class CodeBertLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.codebert = AutoModel.from_pretrained(config.bert_path)
        for param in self.codebert.parameters():  # Selective freezing
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
        初始化检测器
        :param config: 包含模型路径、设备等配置的Config对象
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.bert_path,
            local_files_only=True
        )
        self.model = CodeBertLSTM(config).to(config.device)

    def load_model(self, model_path):
        """
        加载预训练模型
        :param model_path: 模型权重文件路径 (.pt或.bin)
        """
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.config.device)
            )
            self.model.eval()
            print(f"成功加载模型权重: {model_path}")
        except Exception as e:
            raise ValueError(f"模型加载失败: {str(e)}")

    def predict(self, base64_content):
        """
        预测单条样本
        :param base64_content: Base64编码的待检测内容
        :return: 包含预测结果的字典
        """
        try:
            start_time = time.time()

            # 1. 强化解码流程
            cleaned = WebShellPreprocessor.clean_base64(base64_content)
            decoded = base64.b64decode(cleaned).decode('utf-8', errors='ignore')

            # 2. 特征提取（使用预处理器的方法）
            features = WebShellPreprocessor.extract_webshell_features(decoded)

            # 3. Tokenize（添加截断和填充提示）
            encoding = self.tokenizer(
                features,
                max_length=self.config.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )

            # 4. 预测
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
                'error': f"预测失败: {str(e)}",
                'raw_content': base64_content[:100] + '...'  # 返回部分原始内容用于调试
            }


# 4. 数据加载优化
def create_data_loaders(opcode_sequences, labels, tokenizer, config):
    # 使用分层分割
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

    # 动态类别权重
    class_counts = torch.bincount(torch.tensor([y for _, y in train]))
    weights = 1. / class_counts.float()
    samples_weights = weights[torch.tensor([y for _, y in train])]

    # 数据集类
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

    # 创建加载器
    train_loader = DataLoader(
        WebShellDataset(train),
        batch_size=config.batch_size,
        sampler=WeightedRandomSampler(samples_weights, len(samples_weights)),
        num_workers=4
    )
    val_loader = DataLoader(WebShellDataset(val), batch_size=config.batch_size)
    test_loader = DataLoader(WebShellDataset(test), batch_size=config.batch_size)

    return train_loader, val_loader, test_loader


# 5. 训练流程优化
def train_model(model, train_loader, val_loader, optimizer, criterion, config):
    best_f1 = 0.0
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        # 使用更健壮的进度条实现
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

        # 验证阶段
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

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return accuracy, f1, recall, precision


# 主函数
def main():
    # 初始化
    config = Config()
    preprocessor = WebShellPreprocessor()

    # 数据加载与预处理
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

    # 训练流程
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

    # 测试评估
    model.load_state_dict(torch.load('best_codebert_model.pt'))
    test_metrics = evaluate_model(model, test_loader, config)
    print(f"\nTest Metrics - Acc: {test_metrics[0]:.4f} | F1: {test_metrics[1]:.4f} | "
          f"Recall: {test_metrics[2]:.4f} | Precision: {test_metrics[3]:.4f}")

    # 部署检测器
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
        print(f"Sample {i + 1}: {'Malicious' if result['prediction'] else 'Normal'} "
              f"(Confidence: {result['confidence']:.1%})")


if __name__ == '__main__':
    main()