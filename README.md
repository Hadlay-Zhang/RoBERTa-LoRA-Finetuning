# Project 2: BERT Fine-tuning for Text Classification

## Team Members
Yukang Luo (yl13427@nyu.edu), Zhilin Zhang (zz10068@nyu.edu), Yumeng Qian (yq2480@nyu.edu).

## Usage
1. Clone the repo:
```Shell
git clone https://github.com/Hadlay-Zhang/RoBERTa-LoRA-Finetuning.git
```

2. Prepare venv environment
```Shell
python -m venv env
source env/bin/activate
cd RoBERTa-LoRA-Finetuning
pip install -r requirements.txt
```

3. Training and Evaluation
```Shell
python main.py --peft_method lora --output_dir results_lora_qkd_r8_a16_lr2e-5 --target_modules query value attention.output.dense --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 --num_train_epochs 3 --learning_rate 2e-5
python main.py --peft_method loha --output_dir results_loha_qkv_r4_a8_lr2e-5 --target_modules query value key --lora_r 4 --lora_alpha 8 --lora_dropout 0.1 --num_train_epochs 3 --learning_rate 2e-5
python main.py --peft_method lokr --output_dir results_lokr_qkv_r16_a32_lr2e-5 --target_modules query value key --lora_r 16 --lora_alpha 32 --lora_dropout 0.1 --num_train_epochs 3 --learning_rate 2e-5
python main.py --peft_method adalora --output_dir results_adalora_qvd_r4-6_a2_lr5e-5 --target_modules query value attention.output.dense --lora_r 4 --adalora_init_r 6  --lora_alpha 8 --adalora_tinit 0 --adalora_tfinal 0 --adalora_deltaT 1 --adalora_beta1 0.85 --adalora_beta2 0.85 --lora_dropout 0.1 --num_train_epochs 3 --learning_rate 5e-5
```

4. Inference
```Shell
python infer.py --model_dir /path/to/checkpoint --output_csv /path/to/save_dir/prediction.csv --batch_size 128
```

## Possible Issues

### 1. RuntimeError: Failed to find C compiler. Please specify via CC environment variable.

Run following instruction to fix:
```Shell
sudo apt-get update && sudo apt-get install build-essential
```
