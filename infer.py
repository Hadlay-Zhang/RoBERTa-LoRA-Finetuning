import os
import pandas as pd
from transformers import RobertaForSequenceClassification
from peft import PeftModel
import torch
import argparse
from datasets import ClassLabel

import config
import utils
import model as model

def main_infer(args):
    """Main function to run inference on unlabelled data."""
    print("Starting inference process with arguments:")
    print(f"  Model Directory: {args.model_dir}")
    print(f"  Output CSV: {args.output_csv}")
    print(f"Using device: {config.DEVICE}")

    # 1. Load Tokenizer
    tokenizer = utils.load_tokenizer(config.BASE_MODEL)

    # 2. Load Unlabelled Data
    unlabelled_dataset = utils.load_unlabelled_data(config.UNLABELLED_DATA_PATH)
    if unlabelled_dataset is None:
        print("Exiting due to data loading error.")
        return

    # Infer text column if needed
    if config.TEXT_COLUMN not in unlabelled_dataset.column_names:
        print(f"Warning: Column '{config.TEXT_COLUMN}' not found. Trying to infer...")
        if len(unlabelled_dataset.column_names) == 1:
            text_col_to_use = unlabelled_dataset.column_names[0]
            print(f"Using column '{text_col_to_use}' as text column.")
        else:
             print(f"Error: Cannot determine text column in {config.UNLABELLED_DATA_PATH}. Exiting.")
             return
    else:
        text_col_to_use = config.TEXT_COLUMN


    # 3. Load label info (needed for base model init)
    print("Loading label info from training data...")
    temp_dataset = utils.load_data(config.DATASET_NAME, split='train[:1%]') # Load small part
    if isinstance(temp_dataset.features[config.LABEL_COLUMN], ClassLabel):
        num_labels = temp_dataset.features[config.LABEL_COLUMN].num_classes
        class_names = temp_dataset.features[config.LABEL_COLUMN].names
        id2label = {i: label for i, label in enumerate(class_names)}
    else:
        labels = temp_dataset.unique(config.LABEL_COLUMN)
        num_labels = len(labels)
        id2label = {i: str(label) for i, label in enumerate(labels)}


    # 4. Preprocess Unlabelled Data
    print("Preprocessing unlabelled data...")
    unlabelled_dataset = utils.preprocess_dataset_text(unlabelled_dataset, text_col_to_use)

    test_dataset = unlabelled_dataset.map(
        utils.preprocess_data,
        batched=True,
        fn_kwargs={'tokenizer': tokenizer, 'text_column': text_col_to_use},
        remove_columns=unlabelled_dataset.column_names
    )
    print(f"Preprocessed dataset size: {len(test_dataset)}")

    # 5. Load Base Model
    base_model = model.load_base_model(config.BASE_MODEL, num_labels, id2label)

    # 6. Load PEFT Adapter using args.model_dir
    print(f"Loading PEFT adapter from: {args.model_dir}")
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found at {args.model_dir}")
        print("Please provide the correct path to the saved adapter checkpoint directory (e.g., results_run1/final_checkpoint).")
        return

    try:
        # Load the PEFT model using the base model and the adapter directory path
        inference_model = PeftModel.from_pretrained(base_model, args.model_dir)
        inference_model = inference_model.merge_and_unload() # Merge adapter weights for faster inference
        print("PEFT adapter loaded and merged.")
    except Exception as e:
        print(f"Error loading PEFT adapter: {e}")
        print(f"Ensure that a valid adapter checkpoint exists at {args.model_dir}")
        return

    inference_model.to(config.DEVICE)
    # model.print_trainable_parameters(inference_model)

    # 7. Get Data Collator
    data_collator = utils.get_data_collator(tokenizer)

    # 8. Run Inference
    print("Running inference on the unlabelled dataset...")
    predictions = utils.evaluate_model(
        inference_model=inference_model,
        dataset=test_dataset,
        labelled=False, # Set to False for unlabelled data
        batch_size=args.batch_size, # Use batch_size from args for inference
        data_collator=data_collator
    )

    # 9. Format and Save Predictions
    print("Formatting predictions...")
    # Generate IDs or use existing ones if present
    if 'ID' in unlabelled_dataset.column_names:
         output_ids = unlabelled_dataset['ID']
    elif 'id' in unlabelled_dataset.column_names:
         output_ids = unlabelled_dataset['id']
    else:
        output_ids = range(len(predictions))

    df_output = pd.DataFrame({
        'ID': output_ids,
        'Label': predictions.cpu().numpy()
    })

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_csv)
    if output_dir: # Check if output_dir is not empty (e.g., if path is just filename)
        os.makedirs(output_dir, exist_ok=True)

    df_output.to_csv(args.output_csv, index=False)
    print(f"Inference complete. Predictions saved to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a trained PEFT model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory of PEFT adapter checkpoint (e.g., results_run1/final_checkpoint).")
    parser.add_argument("--output_csv", type=str, default=config.DEFAULT_INFERENCE_OUTPUT_CSV, help=f"Path to save the output predictions CSV file (default: {config.DEFAULT_INFERENCE_OUTPUT_CSV}).")
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--use_cleaning", action='store_true')

    args = parser.parse_args()
    main_infer(args)