import os
import pandas as pd
from transformers import TrainingArguments, Trainer, TrainerCallback
from datasets import ClassLabel
import torch
import config
import utils
import math
import model

# callback to save the last checkpoint
class SaveLastCheckpointCallback(TrainerCallback):
    """A TrainerCallback that saves the model checkpoint at the very end of training."""
    def __init__(self, output_dir_name="last_checkpoint"):
        super().__init__()
        self.output_dir_name = output_dir_name

    def on_train_end(self, args: TrainingArguments, state, control, model=None, **kwargs):
        last_checkpoint_dir = os.path.join(args.output_dir, self.output_dir_name)
        if model is not None:
            print(f"\nCallback: Saving final model checkpoint (end of training) to {last_checkpoint_dir}")
            try:
                model.save_pretrained(last_checkpoint_dir)
                print(f"Callback: Final model checkpoint saved successfully to {last_checkpoint_dir}")
            except Exception as e:
                print(f"Callback: Error saving final model checkpoint: {e}")
        else:
             print("\nCallback: Model not provided to on_train_end, cannot save last checkpoint.")

def main_train(args):
    """Main function to run the training process"""
    print(f"Starting training process with PEFT method: {args.peft_method}")
    print("Arguments:")
    print(vars(args)) # print arguments
    print(f"Using device: {config.DEVICE}")

    utils.set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    best_checkpoint_dir = f"{args.output_dir}/best_checkpoint"
    last_checkpoint_folder_name = "last_checkpoint"

    # load Tokenizer and preprocess data
    tokenizer = utils.load_tokenizer(config.BASE_MODEL)
    dataset = utils.load_data(config.DATASET_NAME, split='train')
    dataset = utils.preprocess_dataset_text(dataset, config.TEXT_COLUMN)

    # extract labels' info
    if isinstance(dataset.features[config.LABEL_COLUMN], ClassLabel):
        num_labels = dataset.features[config.LABEL_COLUMN].num_classes
        class_names = dataset.features[config.LABEL_COLUMN].names
        id2label = {i: label for i, label in enumerate(class_names)}
        print(f"Number of labels: {num_labels}")
        print(f"Label names: {class_names}")
    else:
        labels = dataset.unique(config.LABEL_COLUMN)
        num_labels = len(labels)
        id2label = {i: str(label) for i, label in enumerate(labels)}
        print(f"Inferred {num_labels} labels from data.")

    # data mapping
    tokenized_dataset = dataset.map(
        utils.preprocess_data,
        batched=True,
        fn_kwargs={'tokenizer': tokenizer, 'text_column': config.TEXT_COLUMN},
        remove_columns=[config.TEXT_COLUMN]
    )
    if config.LABEL_COLUMN != 'labels':
        tokenized_dataset = tokenized_dataset.rename_column(config.LABEL_COLUMN, "labels")

    # split dataset for evaluation (using same eval size as in the starter code)
    split_datasets = tokenized_dataset.train_test_split(
        test_size=config.TEST_SIZE_EVAL,
        seed=args.seed,
        stratify_by_column="labels"
    )
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['test']
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    num_devices = 1
    effective_train_batch_size = args.train_batch_size * num_devices
    num_training_steps_per_epoch = math.ceil(len(train_dataset) / effective_train_batch_size)
    total_training_steps = num_training_steps_per_epoch * args.num_train_epochs
    print(f"Calculated total training steps: {total_training_steps}")

    # load pretrained RoBERTa model
    base_model = model.load_base_model(config.BASE_MODEL, num_labels, id2label)

    model_log_path = os.path.join(args.output_dir, config.DEFAULT_MODEL_LOG)

    # create PEFT Model
    param_size, peft_model = model.create_peft_model(
        model=base_model,
        args=args,
        total_training_steps=total_training_steps, # AdaLoRA needs specified training steps
    )
    utils.check_param_size(param_size) # check the size of trainable parameters, stop training if exceeds limit

    # setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir, # save dir
        per_device_train_batch_size=args.train_batch_size, # training batch size
        per_device_eval_batch_size=args.eval_batch_size, # evaluation batch size
        learning_rate=args.learning_rate, # learning rate
        num_train_epochs=args.num_train_epochs, # training epochs
        optim=args.optimizer, # optimizer
        logging_dir=f"{args.output_dir}/logs", # logging dir
        logging_strategy="steps",
        logging_steps=args.logging_steps, # logging_steps
        eval_strategy="epoch", # evaluate every epoch
        save_strategy="epoch", # save checkpoint every epoch
        save_total_limit=1,
        load_best_model_at_end=True, # load best model
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS, # number of workers for dataloader
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        use_cpu=config.USE_CPU_TRAINING,
        gradient_checkpointing=False,
    )

    # data collator
    data_collator = utils.get_data_collator(tokenizer)
    save_last_callback = SaveLastCheckpointCallback(output_dir_name=last_checkpoint_folder_name)
    # initialize Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=utils.compute_metrics,
        callbacks=[save_last_callback] # last epoch callback
    )

    # training process
    print(f"Starting PEFT model training using {args.peft_method}...")
    train_result = trainer.train()

    # save metrics and model
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # save the best adapter model checkpoint
    print(f"\nSaving BEST model checkpoint to {best_checkpoint_dir}")
    trainer.save_model(best_checkpoint_dir) # save the adapter config and weights of the best model

    # evaluate the best trained adapter model
    print("\nEvaluating the final best model on the evaluation set...")
    eval_metrics, _ = utils.evaluate_model(
        inference_model=trainer.model,
        dataset=eval_dataset,
        labelled=True,
        batch_size=args.eval_batch_size,
        data_collator=data_collator
    )
    print("Final Evaluation Metrics:", eval_metrics)
    trainer.log_metrics("eval_final", eval_metrics)
    trainer.save_metrics("eval_final", eval_metrics)
    
    # summary of results
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print(f"Validation accuracy: {eval_metrics['accuracy'] * 100:.2f}%")
    print("="*50 + "\n")
    
    return eval_metrics['accuracy'] # Return the evaluation accuracy