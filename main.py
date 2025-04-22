import argparse
import train
import config

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning a RoBERTa model for text classification using LoRA.")

    parser.add_argument("--output_dir", type=str, default=config.DEFAULT_OUTPUT_DIR, help=f"Directory to save checkpoints, logs, and results (default: {config.DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help=f"Random seed for reproducibility (default: {config.DEFAULT_SEED}).")

    # PEFT Method
    parser.add_argument("--peft_method", type=str, default="lora", help="PEFT method to use: lora, loha, lokr, adalora.", choices=['lora', 'loha', 'lokr', 'adalora'])
    parser.add_argument("--target_modules", nargs='+', default=['query', 'value'], help="List of module names to target for PEFT adaptation (e.g., query value key).")
    parser.add_argument("--layers_to_transform", nargs='+', type=int, default=None, help="List of layer indices (integers) to apply LoRA.")

    parser.add_argument("--lora_r", type=int, default=config.DEFAULT_LORA_R, help=f"Rank r for LoRA/LoHa/LoKr, Target r for AdaLoRA (default: {config.DEFAULT_LORA_R}).")
    parser.add_argument("--lora_alpha", type=int, default=config.DEFAULT_LORA_ALPHA, help=f"Alpha parameter for LoRA/LoHa/LoKr (default: {config.DEFAULT_LORA_ALPHA}). Ignored by AdaLoRA.")
    parser.add_argument("--lora_dropout", type=float, default=config.DEFAULT_LORA_DROPOUT, help=f"Dropout probability for LoRA layers (default: {config.DEFAULT_LORA_DROPOUT}). Also used by AdaLoRA.")

    # dropout specific to LoHa/LoKr
    parser.add_argument("--rank_dropout", type=float, default=0.0, help="Rank dropout probability for LoHa/LoKr.")
    parser.add_argument("--module_dropout", type=float, default=0.0, help="Module dropout probability for LoHa/LoKr.")

    # only for AdaLoRA
    parser.add_argument("--adalora_init_r", type=int, default=12, help="Initial rank for AdaLoRA (default: 12).")
    parser.add_argument("--adalora_tinit", type=int, default=0, help="Delay step for rank allocation warmup in AdaLoRA (default: 0).")
    parser.add_argument("--adalora_tfinal", type=int, default=0, help="Stop step for rank allocation in AdaLoRA (default: 0 implies constant allocation).")
    parser.add_argument("--adalora_deltaT", type=int, default=1, help="Step interval for rank allocation update in AdaLoRA (default: 1).")
    parser.add_argument("--adalora_beta1", type=float, default=0.85, help="Hyperparameter beta1 for AdaLoRA ortho reg (default: 0.85).")
    parser.add_argument("--adalora_beta2", type=float, default=0.85, help="Hyperparameter beta2 for AdaLoRA ortho reg (default: 0.85).")

    # Training Hyperparams
    parser.add_argument("--learning_rate", type=float, default=config.DEFAULT_LEARNING_RATE, help=f"Initial learning rate (default: {config.DEFAULT_LEARNING_RATE}).")
    parser.add_argument("--num_train_epochs", type=int, default=config.DEFAULT_NUM_TRAIN_EPOCHS, help=f"Number of training epochs (default: {config.DEFAULT_NUM_TRAIN_EPOCHS}).")
    parser.add_argument("--train_batch_size", type=int, default=config.DEFAULT_TRAIN_BATCH_SIZE, help=f"Batch size per device during training (default: {config.DEFAULT_TRAIN_BATCH_SIZE}).")
    parser.add_argument("--eval_batch_size", type=int, default=config.DEFAULT_EVAL_BATCH_SIZE, help=f"Batch size per device during evaluation (default: {config.DEFAULT_EVAL_BATCH_SIZE}).")
    parser.add_argument("--optimizer", type=str, default=config.DEFAULT_OPTIMIZER, help=f"Optimizer to use (default: {config.DEFAULT_OPTIMIZER}).")

    args = parser.parse_args()

    # call the main training function from train.py
    train.main_train(args) 
    print(f"Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()