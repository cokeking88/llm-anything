# LoRA技术

## LoRA详解

### 什么是LoRA？
**全称**：Low-Rank Adaptation（低秩适配）

**核心思想**：大模型的参数更新是"低秩"的，可以用小矩阵近似

```
原始权重：W (4096×4096) = 1600万参数
LoRA：W + A×B
A (4096×16) + B (16×4096) = 13万参数（仅0.8%）
```

### 类比理解
```
全参数微调：重新装修整个房子
LoRA：只换窗帘和沙发套，效果差不多但省很多钱
```

## LoRA原理

### 低秩假设
**一句话**：矩阵看起来很大，但实际信息量很小

```
一个 1000×1000 的矩阵（100万个数字）
如果秩只有10，只需要20个数字就能表示

低秩分解：1000×10 + 10×1000 = 20,000 参数
压缩了50倍！
```

### 数学原理
```
原始微调：W' = W + ΔW
LoRA：ΔW = A×B，其中A∈R^(d×r), B∈R^(r×k)

关键参数：
  - r：秩，控制参数量
  - d, k：原始矩阵维度
  - 参数量：d×r + r×k = r×(d+k)
```

### 为什么有效？
```
1. 低秩假设：模型更新是低秩的
2. 参数效率：大幅减少训练参数
3. 稳定训练：避免过拟合
4. 即插即用：可合并到原模型
```

## LoRA配置

### 关键参数
```
1. r（秩）：控制参数量和表达能力
   - r=8：参数少，适合简单任务
   - r=16：平衡，常用选择
   - r=32：参数多，适合复杂任务

2. lora_alpha：缩放因子
   - 控制LoRA权重的缩放
   - 通常设为r的2倍

3. target_modules：应用到哪些层
   - q_proj, v_proj：注意力层
   - 全部线性层：更全面的适配

4. lora_dropout：Dropout率
   - 防止过拟合
   - 通常0.05-0.1
```

### 配置示例
```python
from peft import LoraConfig

# 基础配置
lora_config = LoraConfig(
    r=16,                    # 秩
    lora_alpha=32,           # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 应用层
    lora_dropout=0.1         # Dropout
)

# 全面配置
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none"
)
```

## LoRA代码

### 基本实现
```python
from peft import LoraConfig, get_peft_model

# LoRA配置
lora_config = LoraConfig(
    r=16,                    # 秩
    lora_alpha=32,           # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 应用到哪些层
    lora_dropout=0.1
)

# 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4M || all params: 7B || trainable%: 0.06%
```

### 完整训练流程
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. 加载模型
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 配置LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

# 3. 应用LoRA
model = get_peft_model(model, lora_config)

# 4. 查看可训练参数
model.print_trainable_parameters()

# 5. 准备数据（示例）
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

# 6. 训练（简化示例）
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# ... 训练循环 ...

# 7. 保存LoRA权重
model.save_pretrained("./lora_weights")
```

## LoRA变体

### QLoRA
```
概念：4-bit量化 + LoRA
优点：显存需求更低
实现：使用BitsAndBytesConfig
```

### AdaLoRA
```
概念：自适应LoRA
优点：自动调整秩
实现：动态分配参数预算
```

### GLoRA
```
概念：泛化LoRA
优点：更好的泛化能力
实现：更灵活的适配方式
```

### DoRA
```
概念：权重分解LoRA
优点：效果更好
实现：分解权重更新
```

## LoRA最佳实践

### 1. 秩的选择
```
简单任务：r=8-16
复杂任务：r=16-32
资源有限：r=4-8
效果优先：r=32-64
```

### 2. 目标层选择
```
注意力层：q_proj, v_proj（基础选择）
全注意力层：q_proj, k_proj, v_proj, o_proj（全面选择）
全网络：所有线性层（最大适配）
```

### 3. 训练技巧
```
1. 使用低学习率：1e-4到5e-4
2. 渐进式训练：先训练部分层
3. 早停策略：防止过拟合
4. 混合精度：节省显存
```

### 4. 合并权重
```python
# 合并LoRA权重到原模型
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 加载LoRA权重
model = PeftModel.from_pretrained(base_model, "./lora_weights")

# 合并
merged_model = model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("./merged_model")
```

## 实战示例

### 微调LLaMA-7B
```python
# 完整配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    inference_mode=False
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,
    optim="adamw_torch",
    report_to="none"
)
```

## 评估指标

### 参数效率
```
可训练参数占比：0.01%-1%
显存减少：50%-90%
训练速度：比全参数微调快2-5倍
```

### 效果评估
```
任务性能：通常达到全参数微调的90-95%
通用能力：保留较好
过拟合风险：较低
```

## 学习资源
- LoRA论文：Low-Rank Adaptation of Large Language Models
- Hugging Face PEFT文档
- LoRA GitHub仓库

## 下一步
继续学习 [QLoRA技术](03_QLoRA技术.md)