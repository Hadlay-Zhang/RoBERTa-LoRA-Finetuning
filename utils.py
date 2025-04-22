import torch
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset, Dataset, ClassLabel
import pickle
from transformers import RobertaTokenizer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score
import evaluate
from tqdm import tqdm
from torch.utils.data import DataLoader
import re
import nltk
from nltk.corpus import stopwords
import string
import config

# download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_tokenizer(model_name: str):
    """Loads the tokenizer for the specified model."""
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    return tokenizer

def load_data(dataset_name: str, split: str = 'train'):
    """Loads the specified dataset split."""
    print(f"Loading dataset: {dataset_name}, split: {split}")
    dataset = load_dataset(dataset_name, split=split)
    return dataset

def preprocess_data(examples, tokenizer, text_column: str):
    """Tokenizes the text data."""
    return tokenizer(examples[text_column], truncation=True, padding='max_length', max_length=512)

def get_data_collator(tokenizer):
    """Returns a data collator with padding."""
    return DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

def compute_metrics(pred):
    """Computes accuracy metric."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    return {'accuracy': accuracy}

def evaluate_model(inference_model, dataset, labelled=True, batch_size=8, data_collator=None):
    """Evaluate a model on the evaluation set, or infer on unlabelled data."""
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    device = config.DEVICE

    inference_model.to(device)
    inference_model.eval()

    all_predictions = []
    if labelled:
        metric = evaluate.load('accuracy')

    action = "Evaluating" if labelled else "Inferencing"
    print(f"Starting {action} on {len(dataset)} samples...")
    # loop over the DataLoader
    for batch in tqdm(eval_dataloader, desc=action):
        batch_on_device = {}
        input_keys = [k for k in batch.keys() if k != 'labels']

        for k in input_keys:
            batch_on_device[k] = batch[k].to(device)

        if labelled and 'labels' in batch:
             batch_on_device['labels'] = batch['labels'].to(device)

        with torch.no_grad():
            outputs = inference_model(**batch_on_device)


        predictions = outputs.logits.argmax(dim=-1)
        all_predictions.append(predictions.cpu())

        if labelled:
            references = batch["labels"] # references from original batch
            metric.add_batch(predictions=predictions.cpu().numpy(), references=references.cpu().numpy())

    # concat predictions from all batches
    all_predictions = torch.cat(all_predictions, dim=0)

    if labelled:
        eval_metric = metric.compute()
        print(f"Evaluation Metric ({metric.name}): {eval_metric}")
        return eval_metric, all_predictions
    else:
        print("Inference finished.")
        return all_predictions

def load_unlabelled_data(path: str):
    """Loads unlabelled data from a pickle file."""
    print(f"Loading unlabelled data from: {path}")
    try:
        with open(path, "rb") as f:
            loaded_object = pickle.load(f)
        if isinstance(loaded_object, Dataset):
            print("Successfully loaded object as Hugging Face Dataset.")
            return loaded_object
        else:
            print(f"Error: Object loaded from {path} is not a datasets.Dataset. Type found: {type(loaded_object)}")
            print("The code expects 'test_unlabelled.pkl' to contain a Hugging Face Dataset object.")
            return None
    except FileNotFoundError:
        print(f"Error: Unlabelled data file not found at {path}")
        return None
    except Exception as load_e:
        print(f"Error loading or processing pickle file {path}: {load_e}")
        return None

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Set seed to {seed}")

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params} || All params: {all_param} || Trainable %: {100 * trainable_params / all_param:.2f}")
    return trainable_params

def check_param_size(trainable_params, limit=1_000_000):
    """Check param size against the constraint."""
    if trainable_params > limit:
        raise ValueError(f"Trainable parameters ({trainable_params}) exceed the limit of {limit}! Training terminated.")
    else:
        print(f"\nTrainable parameters ({trainable_params}) are within the limit of {limit}.")

def clean_text(text):
    """Clean text by removing HTML tags, URLs, stopwords, and punctuations."""
    # remove HTML tags
    html = re.compile('<.*?>')
    text = html.sub('', text)
    # remove URLs
    url = re.compile(r'https?://\S+|www\.\S+')
    text = url.sub(r'', text)
    # remove stopwords
    stop_words = set(stopwords.words("english"))
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
    # remove punctuation
    punc = string.punctuation
    text = text.translate(str.maketrans('', '', punc))
    return text

def preprocess_dataset_text(dataset, text_column):
    """Apply text cleaning to the entire dataset."""
    def clean_examples(examples):
        examples[text_column] = [clean_text(text) for text in examples[text_column]]
        return examples
    cleaned_dataset = dataset.map(clean_examples, batched=True, desc="Cleaning text data")
    
    print(f"Text cleaning completed for {text_column} column.")
    return cleaned_dataset

def plot_training_curves(output_dir, log_filename="metrics_log.jsonl", plot_filename="training_curves.png"):
    """Reads the metrics log file (from callback) and plots training curves."""
    log_file_path = os.path.join(output_dir, log_filename)
    plot_file_path = os.path.join(output_dir, plot_filename)

    if not os.path.exists(log_file_path):
        print(f"Error: Metrics log file '{log_filename}' not found in {output_dir}")
        return

    # turn into a DataFrame
    metrics_data = []
    try:
        with open(log_file_path, "r") as f:
            for line in f:
                try:
                    metrics_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {log_file_path}: {line.strip()}")
        if not metrics_data:
            print(f"No data found in metrics log file: {log_file_path}")
            return
        df = pd.DataFrame(metrics_data)
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        df.dropna(subset=['epoch'], inplace=True)
        df = df.sort_values(by=["epoch", "step"]).drop_duplicates(subset=['epoch'], keep='last')
    except Exception as e:
        print(f"Error reading or parsing metrics log file {log_file_path}: {e}")
        return

    if df.empty:
        print("No valid metric data found to plot.")
        return

    print(f"Plotting metrics from {len(df)} data points found in {log_filename}...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # 2 rows, 1 column

    # loss
    if 'train_loss' in df.columns and 'eval_loss' in df.columns:
        loss_df = df[['epoch', 'train_loss', 'eval_loss']].dropna()
        if not loss_df.empty:
            axes[0].plot(loss_df['epoch'], loss_df['train_loss'], label='Train Loss (Callback Eval)', marker='o', linestyle='-')
            axes[0].plot(loss_df['epoch'], loss_df['eval_loss'], label='Eval Loss (Callback Eval)', marker='x', linestyle='--')
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training & Evaluation Loss vs. Epochs (Callback Log)")
            axes[0].legend()
            axes[0].grid(True, linestyle=':')
        else:
            print("Warning: No non-null loss data to plot.")

    # accuracy
    if 'train_accuracy' in df.columns and 'eval_accuracy' in df.columns:
        acc_df = df[['epoch', 'train_accuracy', 'eval_accuracy']].dropna()
        if not acc_df.empty:
            axes[1].plot(acc_df['epoch'], acc_df['train_accuracy'], label='Train Acc (Callback Eval)', marker='o', linestyle='-')
            axes[1].plot(acc_df['epoch'], acc_df['eval_accuracy'], label='Eval Acc (Callback Eval)', marker='x', linestyle='--')
            axes[1].set_xlabel("Epochs")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title("Training & Evaluation Accuracy vs. Epochs (Callback Log)")
            axes[1].legend()
            axes[1].grid(True, linestyle=':')
            axes[1].set_ylim(bottom=max(0, acc_df[['train_accuracy', 'eval_accuracy']].min().min() - 0.05),
                             top=min(1.05, acc_df[['train_accuracy', 'eval_accuracy']].max().max() + 0.05)) # Adjust y-limits reasonably
        else:
            print("Warning: No non-null accuracy data to plot.")

    plt.tight_layout()
    try:
        plt.savefig(plot_file_path)
        print(f"Training curves plot saved to {plot_file_path}")
    except Exception as e:
        print(f"Error saving plot to {plot_file_path}: {e}")
    plt.close(fig)