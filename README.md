# CodeBERT-LSTM WebShell Detector

This repository contains a **WebShell detection framework** based on [CodeBERT](https://huggingface.co/microsoft/codebert-base) with a **bi-directional LSTM + Attention layer**.  
It provides **end-to-end training, evaluation, and prediction** using WebShell datasets.

---

## Features
- **Data Preprocessing**: Cleans base64-encoded data and extracts suspicious features (eval, system, SQLi, XSS, etc.).
- **Model Architecture**:  
  - Pretrained CodeBERT embeddings (frozen).  
  - Bi-directional 2-layer LSTM with dropout.  
  - Attention mechanism for feature weighting.  
  - Fully connected classifier with ReLU activation.
- **Training Pipeline**: Supports stratified train/val/test split, class balancing with `WeightedRandomSampler`, gradient clipping, and checkpoint saving.
- **Evaluation Metrics**: Accuracy, F1, Recall, Precision, Confusion Matrix.
- **Deployment-ready**: Provides a `WebShellDetector` class for single-sample prediction.

---

## Project Structure
```text
codebert-training/
├── train_codebert.py       # Main training script (this file)
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md
```
---

## Installation
```bash
git clone https://github.com/<your-username>/codebert-training.git
cd codebert-training
pip install -r requirements.txt
```
---

## Usage

### 1. Prepare Dataset
Default paths in the script:  
```python
df = preprocessor.load_and_clean_data([
    'webshell.csv',
    'webshell_data.csv'
])
```

---

### 2. Train Model
```bash
python train_codebert.py
```
- Trains model for `epochs=3` (adjustable in `Config`).  
- Saves the best model to `best_codebert_model.pt`.

---

### 3. Evaluate Model
```bash
python train_codebert.py
```

At the end of training, the script evaluates on the test set:  
```yaml
Test Metrics - Acc: 0.9821 | F1: 0.9756 | Recall: 0.9712 | Precision: 0.9801
```

---

### 4. Predict Samples
The script includes a demo for sample predictions:  
```python
test_samples = [
    "PD9waHAgZXZhbCgkX1BPU1RbJ2NtZCddKTs=",  # PHP shell
    "U0hFTEwgY21kLmV4ZQ==",                  # Suspicious
    "aW5kZXguaHRtbA=="                      # Normal
]

detector = WebShellDetector(config)
detector.load_model("best_codebert_model.pt")

for sample in test_samples:
    print(detector.predict(sample))
```

**Output:**  
```less
Sample 1: Malicious (Confidence: 98.7%)
Sample 2: Malicious (Confidence: 94.1%)
Sample 3: Normal (Confidence: 99.2%)
```

---

## Dependencies
- torch  
- transformers  
- scikit-learn  
- pandas  
- tqdm  

GPU recommended, with CUDA device set via:  
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

Install all dependencies:  
```bash
pip install -r requirements.txt
```

---

## Data & Model Files
- Model weights (`*.pt`) are **not tracked in GitHub**.  
- Store them in **Git LFS** or external storage (Google Drive, Baidu Netdisk).  
- Update this README with download links if sharing.
