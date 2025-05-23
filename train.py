import os
import pandas as pd
from transformers import TrainingArguments, Trainer, TrainerCallback, TrainerState, TrainerControl
from datasets import ClassLabel
import torch
import json
import config
import utils
import math
import model
from collections import defaultdict

# callback to save metrics within epoch
class MetricsLoggerCallback(TrainerCallback):
    """Logs metrics from Trainer's `on_log`."""
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "metrics_log.jsonl")
        self.step_metrics_buffer = defaultdict(dict)
        self.last_written_step = -1 # Track the last step actually written

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self.step_metrics_buffer.clear()
        self.last_written_step = -1

    def _write_log(self, step_to_write, state):
        if step_to_write in self.step_metrics_buffer and step_to_write > self.last_written_step:
            record_data = self.step_metrics_buffer[step_to_write]
            current_epoch = round(record_data.get('epoch', state.epoch if state.epoch is not None else 0), 4)

            final_record = {
                "epoch": current_epoch,
                "step": step_to_write,
                "eval_loss": round(val, 6) if (val := record_data.get("eval_loss")) is not None else None,
                "eval_accuracy": round(val, 6) if (val := record_data.get("eval_accuracy")) is not None else None,
                "train_loss": round(val, 6) if (val := record_data.get("loss", record_data.get("train_loss"))) is not None else None,
                "train_accuracy": round(val, 6) if (val := record_data.get("accuracy", record_data.get("train_accuracy"))) is not None else None,
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(final_record) + "\n")
            self.last_written_step = step_to_write
            # Clean up buffer for steps already written
            keys_to_delete = [k for k in self.step_metrics_buffer if k <= step_to_write]
            for key in keys_to_delete:
                del self.step_metrics_buffer[key]

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        current_step = state.global_step
        if current_step > self.last_written_step + 1:
             for step_to_check in range(self.last_written_step + 1, current_step):
                  if step_to_check in self.step_metrics_buffer:
                       self._write_log(step_to_check, state) # Pass state for epoch fallback

        valid_logs = {k: v for k, v in logs.items() if v is not None} # Avoid overwriting with None
        self.step_metrics_buffer[current_step].update(valid_logs)
        if "eval_loss" in logs:
            self._write_log(current_step, state)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
         """Ensures logs for the final steps are written when training ends."""
         latest_step_in_buffer = max(self.step_metrics_buffer.keys()) if self.step_metrics_buffer else -1
         for step_to_log in range(self.last_written_step + 1, latest_step_in_buffer + 1):
              if step_to_log in self.step_metrics_buffer:
                   self._write_log(step_to_log, state)

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
    param_size, peft_model = model.create_peft_model(
        model=base_model,
        args=args,
        total_training_steps=total_training_steps,
    )
    utils.check_param_size(param_size)

    # instantiate callbacks
    save_last_callback = SaveLastCheckpointCallback(output_dir_name=last_checkpoint_folder_name)
    metrics_logger_callback = MetricsLoggerCallback(output_dir=args.output_dir) # NEW
    steps = num_training_steps_per_epoch // 4 # log every 1/4 of an epoch
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
        logging_steps=steps,     
        eval_strategy="steps",
        eval_steps=steps,                
        save_strategy="steps", 
        save_steps=steps,               
        save_total_limit=1,                   
        load_best_model_at_end=True,          
        metric_for_best_model="accuracy",     
        greater_is_better=True,              
        report_to="none",
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        use_cpu=config.USE_CPU_TRAINING,
        gradient_checkpointing=False,
    )

    # initialize Trainer and data collator
    data_collator = utils.get_data_collator(tokenizer)
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=utils.compute_metrics,
        callbacks=[save_last_callback, metrics_logger_callback] # add metrics_logger_callback
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
    print(f"\nSaving BEST model checkpoint identified by Trainer to {best_checkpoint_dir}")
    trainer.save_model(best_checkpoint_dir) # save the adapter config and weights of the best model

    print("\nEvaluating the final best model on the evaluation set...")
    # get final model, which should be the best one
    final_model_to_eval = trainer.model
    eval_metrics, _ = utils.evaluate_model(
        inference_model=final_model_to_eval,
        dataset=eval_dataset,
        labelled=True,
        batch_size=args.eval_batch_size,
        data_collator=data_collator
    )
    print("Final Evaluation Metrics (Best Model):", eval_metrics)
    trainer.log_metrics("eval_final_best", eval_metrics)
    trainer.save_metrics("eval_final_best", eval_metrics)

    # plotting
    print("\nGenerating curves plot from callback logs...")
    utils.plot_training_curves(args.output_dir) # plot data from metrics_log.jsonl
    print("Plot generation complete.")

    # summary
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print(f"Best model validation accuracy (evaluated at end): {eval_metrics['accuracy'] * 100:.2f}%")
    print(f"Best model checkpoint saved to: {best_checkpoint_dir}")
    print(f"Fractional epoch metrics logged to: {os.path.join(args.output_dir, 'metrics_log.jsonl')}")
    print(f"Training curves plot saved to: {os.path.join(args.output_dir, 'training_curves.png')}")
    print("="*50 + "\n")

    return eval_metrics['accuracy']