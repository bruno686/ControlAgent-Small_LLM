import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 1. 加载模型和分词器
model_name = "../LLMs/qwen2.5-1.5b-instruct"  # 你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. 加载和预处理数据集
dataset = load_dataset('json', data_files='trajectory_gpt.jsonl')

# 拆分数据集为训练集和验证集（80% 训练，20% 验证）
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

def preprocess_function(examples):
    inputs = [f"### 问题: {prompt}\n### 回答: {completion}" for prompt, completion in zip(examples['prompt'], examples['completion'])]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()  # 自回归任务，labels与input_ids相同
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['prompt', 'completion'])

# 3. 设置训练参数
training_args = TrainingArguments(
    output_dir="./control_system_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=5,  
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=1, 
    save_strategy="epoch",  # 修改为与 eval_strategy 一致
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=5,  
    learning_rate=2e-5,  
    weight_decay=0.01,  
    fp16=True,  # 加速训练
    eval_strategy="epoch",  # 更新为新参数名，每个 epoch 评估一次
    per_device_eval_batch_size=4,  # 验证集批大小
    load_best_model_at_end=True,  # 训练结束后加载最佳模型
    metric_for_best_model="loss",  # 以验证集损失作为最佳模型标准
    greater_is_better=False,  # 损失越小越好
)

# 4. 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],  # 添加验证集
    callbacks=None
)

# 5. 开始微调
trainer.train()

# 6. 保存微调后的模型
model.save_pretrained("./control_system_finetuned_model")
tokenizer.save_pretrained("./control_system_finetuned_model")

print("微调完成！模型已保存至 './control_system_finetuned_model'")