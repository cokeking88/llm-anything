# QLoRA技术

## QLoRA详解

### 什么是QLoRA？
**一句话**：4-bit量化 + LoRA，显存需求更低

**效果**：
```
LLaMA-7B：
- 全精度：14GB显存
- QLoRA：5GB显存（可训练）
```

### QLoRA vs LoRA
```
LoRA：
  - 使用FP16/BF16
  - 显存需求：中等
  - 训练速度：快

QLoRA：
  - 使用INT4量化
  - 显存需求：极低
  - 训练速度：较慢（量化/反量化开销）
```

## 4-bit量化

### 量化原理
```
原始：FP32（32位浮点数）
量化：FP16/BF16（16位）→ INT8（8位）→ INT4（4位）

每个参数：
  FP32：4字节
  FP16：2字节
  INT8：1字节
  INT4：0.5字节
```

### 量化类型
```
1. NF4（Normal Float 4-bit）
   - QLoRA使用
   - 正态分布优化
   - 效果最好

2. FP4
   - 浮点4-bit
   - 精度较高

3. INT4
   - 整数4-bit
   - 硬件支持好
```

### 量化效果
```
7B模型：
  - FP16：14GB显存
  - INT8：7GB显存
  - INT4：3.5GB显存

速度提升：
  - INT4推理：比FP16快2-3倍
  - QLoRA训练：比LoRA节省50%显存
```

## QLoRA代码

### 基本实现
```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config
)

# 应用LoRA
model = get_peft_model(model, lora_config)
```

### 完整训练流程
```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # 双重量化
)

# 2. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. 准备模型用于训练
model = prepare_model_for_kbit_training(model)

# 4. 配置LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. 应用LoRA
model = get_peft_model(model, lora_config)

# 6. 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 7. 训练参数
training_args = TrainingArguments(
    output_dir="./qlora_results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch"
)

# 8. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# 9. 保存
model.save_pretrained("./qlora_weights")
```

## QLoRA优化技术

### 双重量化（Double Quantization）
```
概念：对量化参数再进行量化
效果：进一步节省显存
实现：bnb_4bit_use_double_quant=True
```

### 分页优化器（Paged Optimizer）
```
概念：优化器状态分页到CPU
效果：减少GPU显存占用
实现：使用PagedAdamW
```

### 梯度检查点（Gradient Checkpointing）
```
概念：用计算换内存
效果：减少激活值显存
实现：gradient_checkpointing_enable()
```

## QLoRA最佳实践

### 1. 量化配置
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # 使用NF4
    bnb_4bit_compute_dtype=torch.float16,  # 计算精度
    bnb_4bit_use_double_quant=True  # 双重量化
)
```

### 2. LoRA配置
```python
lora_config = LoraConfig(
    r=16,  # 秩
    lora_alpha=32,  # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 目标层
    lora_dropout=0.1,  # Dropout
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 3. 训练技巧
```
1. 使用混合精度：fp16=True
2. 梯度累积：增加有效批次大小
3. 学习率调度：使用warmup
4. 早停策略：防止过拟合
5. 定期评估：监控验证集性能
```

### 4. 显存优化
```
1. 使用gradient_checkpointing
2. 使用PagedOptimizer
3. 减少batch_size
4. 增加gradient_accumulation_steps
5. 使用更小的序列长度
```

## QLoRA实战示例

### 微调LLaMA-2-7B
```python
# 显存需求：约5-6GB

# 配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none"
)

# 训练
training_args = TrainingArguments(
    output_dir="./llama2-7b-qlora",
    num_train_epochs=5,
    per_device_train_batch_size=2,  # 小批次
    gradient_accumulation_steps=8,  # 梯度累积
    learning_rate=1e-4,
    warmup_steps=100,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)
```

### 微调Qwen-7B
```python
# 显存需求：约6-7GB

# 配置
model_name = "Qwen/Qwen-7B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj", "w1", "w2"],  # Qwen特定层
    lora_dropout=0.1,
    bias="none"
)
```

## QLoRA vs 其他方法

### 显存对比
```
LLaMA-7B模型：
  - FP16全参数：14GB
  - LoRA (FP16)：7-8GB
  - QLoRA (INT4)：5-6GB
  - INT8量化：7GB
```

### 效果对比
```
任务性能（相对全参数微调）：
  - LoRA (FP16)：95-98%
  - QLoRA (INT4)：90-95%
  - INT8量化：85-90%
```

### 适用场景
```
选择QLoRA：
  - 显存有限（如RTX 3090/4090）
  - 训练大模型（7B+）
  - 资源受限环境

选择LoRA：
  - 显存充足
  - 追求最佳效果
  - 训练小模型
```

## 故障排除

### 常见问题
```
1. 显存不足
   解决：减小batch_size，增加gradient_accumulation_steps

2. 训练速度慢
   解决：使用更小的r，减少目标层

3. 效果不佳
   解决：增加训练数据，调整超参数

4. 量化精度损失
   解决：使用FP8量化，或增加训练轮数
```

## 学习资源
- QLoRA论文：Efficient Finetuning of Quantized LLMs
- Hugging Face PEFT文档
- BitsAndBytes文档

## 下一步
继续学习 [训练流程](04_训练流程.md)