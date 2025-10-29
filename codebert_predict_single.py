import os
import torch
import re
from transformers import AutoTokenizer, AutoModel
import argparse
import json


# Must copy MultiTaskBERT class from training code to ensure identical structure
class MultiTaskBERT(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model
        hidden_size = base_model.config.hidden_size

        # Shared layer (must match training definition)
        self.shared_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4)
        )

        # Task-specific heads (must match training definition)
        self.xss_head = torch.nn.Linear(256, 2)       # XSS task: 0=benign, 1=malicious
        self.webshell_head = torch.nn.Linear(256, 2)  # Webshell task: 0=benign, 1=malicious
        self.task_router = torch.nn.Linear(hidden_size, 2)  # Task router (decides XSS vs Webshell likelihood)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # Hidden state of [CLS] token
        features = self.shared_layer(pooled)

        # Task routing probabilities (XSS vs Webshell)
        task_probs = torch.softmax(self.task_router(pooled), dim=1)  # [batch, 2], 0=XSS, 1=Webshell

        return {
            'task_probs': task_probs,
            'xss_logits': self.xss_head(features),   # [batch, 2]
            'ws_logits': self.webshell_head(features)  # [batch, 2]
        }


class PHPCodeParser:
    def __init__(self, model_path, base_model_path):
        """
        Initialize parser for multi-task model
        :param model_path: path to fine-tuned MultiTaskBERT weights (.pt file)
        :param base_model_path: path to base model (e.g., codebert-small)
        """
        # Device setup
        self.device = "cpu"

        # Load tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base_model = AutoModel.from_pretrained(base_model_path)

        # Initialize MultiTask model and load weights
        self.model = MultiTaskBERT(self.base_model)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # Unified English label mapping
        self.label_mapping = {
            0: "xss_benign",
            1: "xss_malicious",
            2: "ws_benign",
            3: "ws_malicious"
        }

    def parse_php_file(self, file_path):
        """Parse a single PHP file, return prediction results for 4 categories"""
        try:
            # Attempt multiple encodings for file reading
            encodings = ['utf-8', 'gbk', 'latin-1', 'utf-16']
            code = None
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        code = f.read()
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            if code is None:
                raise ValueError(f"Failed to decode file (attempted encodings: {encodings})")

            # Preprocess (must match training procedure)
            code = self.preprocess_code(code)

            # Encode for model input
            inputs = self.tokenizer(
                code,
                padding=True,
                truncation=True,
                max_length=128,  # must match training max_len
                return_tensors="pt"
            ).to(self.device)

            # Model inference
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Task routing probabilities
                task_probs = outputs['task_probs'][0]  # [2], 0=XSS, 1=Webshell
                xss_prob = task_probs[0].item()
                ws_prob = task_probs[1].item()

                # Task-specific predictions
                xss_logits = outputs['xss_logits'][0]  # [2]
                ws_logits = outputs['ws_logits'][0]    # [2]
                xss_probs = torch.softmax(xss_logits, dim=0)
                ws_probs = torch.softmax(ws_logits, dim=0)

            # Merge into probabilities for 4 labels
            merged_probs = {
                0: xss_prob * xss_probs[0].item(),  # xss_benign
                1: xss_prob * xss_probs[1].item(),  # xss_malicious
                2: ws_prob * ws_probs[0].item(),    # ws_benign
                3: ws_prob * ws_probs[1].item()     # ws_malicious
            }

            # Select final prediction
            prediction_id = max(merged_probs, key=merged_probs.get)
            confidence = merged_probs[prediction_id]

            # Build result
            result = {
                "file_path": file_path,
                "prediction_id": prediction_id,
                "prediction_label": self.label_mapping[prediction_id],
                "confidence": confidence,
                "task_probabilities": {
                    "xss_task": xss_prob,
                    "webshell_task": ws_prob
                },
                "all_probabilities": {
                    self.label_mapping[id]: prob for id, prob in merged_probs.items()
                }
            }
            return result

        except Exception as e:
            print(f"Error parsing file {file_path}: {str(e)}")
            return None

    def preprocess_code(self, code):
        """Preprocess code (must match training procedure)"""
        code = self.remove_comments(code)
        code_lines = [line.strip() for line in code.splitlines() if line.strip()]
        return "\n".join(code_lines)

    def remove_comments(self, code):
        """Remove PHP comments"""
        code = re.sub(r'//.*?(?=\n|$)', '', code)  # single-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # multi-line comments
        return code

    def parse_php_files_in_directory(self, directory):
        """Batch-parse all PHP files in a directory"""
        if not os.path.isdir(directory):
            raise ValueError(f"Path '{directory}' is not a valid directory")

        results = []
        total_samples = 0

        # Initialize label counts
        label_counts = {
            0: 0,  # xss_benign
            1: 0,  # xss_malicious
            2: 0,  # ws_benign
            3: 0   # ws_malicious
        }

        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.php'):
                    total_samples += 1
                    file_path = os.path.join(root, file)
                    print(f"Parsing: {file_path}")
                    result = self.parse_php_file(file_path)
                    if result:
                        results.append(result)
                        label_counts[result["prediction_id"]] += 1

        if total_samples == 0:
            print(f"Warning: No PHP files found in directory '{directory}'")

        # Compile statistics
        statistics = {
            "total_samples": total_samples,
            "label_statistics": {
                self.label_mapping[label_id]: {
                    "count": count,
                    "percentage": f"{count/total_samples*100:.2f}%" if total_samples > 0 else "0.00%"
                }
                for label_id, count in label_counts.items()
            }
        }

        return results, statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multi-task CodeBERT model for PHP file detection (supports XSS and Webshell)'
    )
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to fine-tuned MultiTaskBERT weights (.pt file)')
    parser.add_argument('--base_model_path', type=str, required=True,
                      help='Path to base model directory (e.g., codebert-small)')
    parser.add_argument('--dir_path', type=str, required=True,
                      help='Directory containing PHP files')
    parser.add_argument('--output', type=str, default='multi_task_detection_results.json',
                      help='Output result file (JSON format)')
    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.dir_path):
        print(f"Error: Directory '{args.dir_path}' does not exist")
        exit(1)

    # Run detection
    try:
        code_parser = PHPCodeParser(args.model_path, args.base_model_path)
        results, statistics = code_parser.parse_php_files_in_directory(args.dir_path)

        print("\n===== Statistics =====")
        print(f"Total samples: {statistics['total_samples']}")
        for label_name, stats in statistics["label_statistics"].items():
            print(f"{label_name}: {stats['count']} ({stats['percentage']})")

        # Save results to JSON
        output_data = {
            "statistics": statistics,
            "results": results
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"Execution failed: {str(e)}")
        exit(1)
