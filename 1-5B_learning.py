import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# 1. 加载模型和分词器
model_name = "../LLMs/qwen2.5-1.5b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    use_rslora=True,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. 加载和预处理数据集
dataset = load_dataset('json', data_files='auto_control_datasets.jsonl')
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

def preprocess_function(examples):
    inputs = [f"### Question: {prompt}\n### Answer: {completion}" for prompt, completion in zip(examples['prompt'], examples['completion'])]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['prompt', 'completion'])

# 4. 设置训练参数
training_args = TrainingArguments(
    output_dir="./control_system_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    save_strategy="epoch",
    save_total_limit=5,
    logging_dir='./logs',
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    eval_strategy="epoch",
    per_device_eval_batch_size=4,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
)

# 5. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
)

# 6. 开始微调
trainer.train()

# 7. 合并 LoRA 权重到基础模型
best_checkpoint = trainer.state.best_model_checkpoint
if best_checkpoint:
    print(f"Loading best checkpoint from: {best_checkpoint}")
    # 重新加载基础模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # 加载 LoRA checkpoint
    model = PeftModel.from_pretrained(model, best_checkpoint)
    # 合并权重
    merged_model = model.merge_and_unload()
else:
    print("No best checkpoint found, merging final model state.")
    merged_model = model.merge_and_unload()

# 8. 保存合并后的全参数模型
merged_model.save_pretrained("./control_system_finetuned_model_merged")
tokenizer.save_pretrained("./control_system_finetuned_model_merged")

print("LoRA 微调完成并合并！完整模型已保存至 './control_system_finetuned_model_merged'")