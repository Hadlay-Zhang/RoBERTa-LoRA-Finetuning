from transformers import RobertaForSequenceClassification
from peft import (
    LoraConfig, # for LoRA
    LoHaConfig, # for LoHa
    LoKrConfig, # for LoKr
    AdaLoraConfig, # for AdaLoRA
    get_peft_model,
    TaskType
)
import config
from utils import print_trainable_parameters

def load_base_model(model_name: str, num_labels: int, id2label: dict):
    """Loads the base sequence classification model."""
    print(f"Loading base model: {model_name} for {num_labels} labels.")
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()} # also pass label2id
    )
    return model

def create_peft_model(model, args, total_training_steps=None):
    """Creates the PEFT model with the specified configuration (LoRA, LoHA, or LoKr)."""
    peft_method = args.peft_method
    print(f"Creating PEFT model using method: {peft_method}")
    peft_config = None

    if peft_method == "lora":
        print(f"  Configuring LoRA with: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        peft_config = LoraConfig(
            task_type=config.TASK_TYPE,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias=config.LORA_BIAS,
        )
    elif peft_method == "loha":
        print(f"  Configuring LoHa with: r={args.lora_r}, alpha={args.lora_alpha}, rank_dropout={args.rank_dropout}, module_dropout={args.module_dropout}")
        peft_config = LoHaConfig(
            task_type=config.TASK_TYPE,
            r=args.lora_r,
            alpha=args.lora_alpha,
            rank_dropout=args.rank_dropout,
            module_dropout=args.module_dropout,
            target_modules=args.target_modules,
        )
    elif peft_method == "lokr":
        print(f"  Configuring LoKr with: r={args.lora_r}, alpha={args.lora_alpha}, rank_dropout={args.rank_dropout}, module_dropout={args.module_dropout}")
        peft_config = LoKrConfig(
            task_type=config.TASK_TYPE,
            r=args.lora_r,
            alpha=args.lora_alpha,
            rank_dropout=args.rank_dropout,
            module_dropout=args.module_dropout,
            target_modules=args.target_modules,
        )
    elif peft_method == "adalora":
        print(f"  Configuring AdaLoRA with: target_r={args.lora_r}, init_r={args.adalora_init_r}, dropout={args.lora_dropout}, tinit={args.adalora_tinit}, tfinal={args.adalora_tfinal}, deltaT={args.adalora_deltaT}, beta1={args.adalora_beta1}, beta2={args.adalora_beta2}")
        peft_config = AdaLoraConfig(
            task_type=config.TASK_TYPE,
            target_r=args.lora_r, # use the general 'r' argument as the target rank
            init_r=args.adalora_init_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            tinit=args.adalora_tinit,
            tfinal=args.adalora_tfinal,
            deltaT=args.adalora_deltaT,
            beta1=args.adalora_beta1,
            beta2=args.adalora_beta2,
            target_modules=args.target_modules,
            total_step=total_training_steps
        )
    else:
        raise ValueError(f"Unsupported PEFT method: {peft_method}. Choose from 'lora', 'loha', 'lokr' and 'adalora'.")

    peft_model = get_peft_model(model, peft_config)
    print(f"PEFT model created with {peft_method.upper()} config.")
    print(f"  Target modules: {args.target_modules}") # log target modules
    param_size = print_trainable_parameters(peft_model) # check for parameter count
    return param_size, peft_model