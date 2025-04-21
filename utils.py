import torch
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