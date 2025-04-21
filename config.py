import torch

# Model & Data Constants
BASE_MODEL = 'roberta-base'
DATASET_NAME = 'ag_news'
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'
UNLABELLED_DATA_PATH = "datasets/test_unlabelled.pkl" # unlabelled data

# LoRA Specific Constants
LORA_BIAS = "none"
TASK_TYPE = "SEQ_CLS" # for sequence classification

# General Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CPU_TRAINING = not torch.cuda.is_available()
DATALOADER_NUM_WORKERS = 16
TEST_SIZE_EVAL = 640 # Size of the validation set split from training data
DEFAULT_OUTPUT_DIR = "results_default" # Default if not specified via CLI
DEFAULT_INFERENCE_OUTPUT_CSV = f"{DEFAULT_OUTPUT_DIR}/inference.csv" # Default inference output path
DEFAULT_MODEL_LOG = "model.log"

# Default values for argparse in main.py (can be overridden)
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_NUM_TRAIN_EPOCHS = 3
DEFAULT_TRAIN_BATCH_SIZE = 128
DEFAULT_EVAL_BATCH_SIZE = 128
DEFAULT_SEED = 42
DEFAULT_OPTIMIZER = "adamw_torch"
DEFAULT_LOGGING_STEPS = 100