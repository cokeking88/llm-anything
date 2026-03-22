# MoE架构

## 什么是MoE？

### 一句话定义
**全称**：Mixture of Experts（混合专家）

**一句话**：每次只激活一部分参数

### 类比理解
```
传统模型：每次用所有参数
MoE模型：根据输入选择最相关的"专家"

像医院：不同病找不同科室的医生
像公司：不同问题找不同专家处理
```

## MoE原理

### 核心思想
```
1. 专家网络：多个专门的子网络
2. 门控网络：决定激活哪些专家
3. 稀疏激活：只激活少数专家
4. 负载均衡：确保专家均衡使用
```

### 架构设计
```
┌─────────────────────────────────────┐
│            MoE Layer                 │
├─────────────────────────────────────┤
│     Input → 门控网络 → 路由决策        │
│                ↓                     │
│    ┌─────────┬─────────┬─────────┐  │
│    │ Expert1 │ Expert2 │ Expert3 │  │
│    │ (激活)   │ (激活)   │ (未激活) │  │
│    └─────────┴─────────┴─────────┘  │
│                ↓                     │
│            加权输出                   │
└─────────────────────────────────────┘
```

### 工作流程
```
输入：x

1. 门控网络计算：
   G(x) = softmax(TopK(x·W_g))

2. 选择Top-K专家：
   激活K个最相关的专家（通常K=2）

3. 专家计算：
   y_i = Expert_i(x)

4. 加权求和：
   y = Σ G_i(x) · y_i
```

## 门控网络

### Top-K门控
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKGate(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
    
    def forward(self, x):
        # 计算门控分数
        logits = self.gate(x)
        
        # Top-K选择
        top_k_values, top_k_indices = torch.topk(
            logits, self.top_k, dim=-1
        )
        
        # Softmax归一化
        top_k_gates = F.softmax(top_k_values, dim=-1)
        
        return top_k_gates, top_k_indices
```

### 负载均衡
```python
class LoadBalancedGate(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, 
                 balance_coeff=0.01):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
        self.balance_coeff = balance_coeff
        self.num_experts = num_experts
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 计算门控分数
        logits = self.gate(x)
        
        # Top-K选择
        top_k_values, top_k_indices = torch.topk(
            logits, self.top_k, dim=-1
        )
        top_k_gates = F.softmax(top_k_values, dim=-1)
        
        # 计算负载均衡损失
        # 统计每个专家被选择的次数
        expert_counts = torch.zeros(self.num_experts).to(x.device)
        for idx in top_k_indices.flatten():
            expert_counts[idx] += 1
        
        # 计算均匀分布
        uniform_dist = torch.ones(self.num_experts) / self.num_experts
        
        # 计算实际分布
        actual_dist = expert_counts / expert_counts.sum()
        
        # 负载均衡损失
        load_balance_loss = F.kl_div(
            actual_dist.log(), 
            uniform_dist,
            reduction='sum'
        ) * self.balance_coeff
        
        return top_k_gates, top_k_indices, load_balance_loss
```

## 专家网络

### 专家设计
```python
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_experts=8, top_k=2):
        super().__init__()
        
        # 创建专家
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = TopKGate(input_dim, num_experts, top_k)
        self.top_k = top_k
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, x.size(-1))
        
        # 获取门控分数和专家索引
        gates, indices = self.gate(x_flat)
        
        # 初始化输出
        output = torch.zeros_like(x_flat)
        
        # 计算每个专家的贡献
        for i in range(self.top_k):
            expert_indices = indices[:, i]
            gate_values = gates[:, i].unsqueeze(-1)
            
            # 对每个专家
            for expert_idx in range(len(self.experts)):
                # 找到选择该专家的token
                mask = (expert_indices == expert_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    output[mask] += gate_values[mask] * expert_output
        
        return output.view(batch_size, seq_len, -1)
```

## 代表模型

### Mixtral 8x7B
```
公司：Mistral AI
发布时间：2023年12月
总参数：46.7B
激活参数：12.9B
专家数：8
每层激活专家：2

特点：
  - 性能接近Llama-2 70B
  - 推理速度与7B模型相当
  - 开源可商用
```

### DeepSeek-V2
```
公司：深度求索
发布时间：2024年5月
总参数：236B
激活参数：21B
专家数：160
每层激活专家：6

特点：
  - 性价比极高
  - 支持128K上下文
  - 中文能力强
```

### GPT-4（疑似MoE）
```
公司：OpenAI
推测架构：MoE
推测参数：约1.8T
推测激活参数：约280B

特点：
  - 最强能力
  - 多模态支持
  - 闭源
```

## MoE优势

### 1. 参数效率
```
优势：总参数多，激活参数少
效果：能力强，速度快

示例：
  Mixtral 8x7B：
  - 总参数：46.7B（接近70B模型能力）
  - 激活参数：12.9B（与7B模型速度相当）
```

### 2. 训练效率
```
优势：每个专家专注于部分数据
效果：训练更稳定，收敛更快

原因：
  - 专家分工明确
  - 减少梯度冲突
  - 提高学习效率
```

### 3. 推理效率
```
优势：稀疏激活，计算量小
效果：推理速度快

对比：
  传统70B模型：每次用70B参数
  MoE 8x7B：每次用13B参数
  速度提升：约5倍
```

## 训练挑战

### 1. 专家坍塌
```
问题：所有输入都路由到少数专家
解决：
  - 负载均衡损失
  - 辅助损失函数
  - 门控正则化
```

### 2. 通信开销
```
问题：专家分布在不同设备
解决：
  - 专家并行
  - 通信优化
  - 缓存机制
```

### 3. 显存占用
```
问题：需要存储所有专家参数
解决：
  - 专家卸载
  - 参数共享
  - 量化压缩
```

## 实现示例

### 完整MoE模型
```python
class MoETransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers,
                 d_ff, num_experts=8, top_k=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # MoE层
        self.layers = nn.ModuleList([
            MoELayer(d_model, d_ff, d_model, num_experts, top_k)
            for _ in range(n_layers)
        ])
        
        # 注意力层（交替使用MoE和注意力）
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        
        for moe_layer, attn_layer in zip(self.layers, self.attention_layers):
            # MoE层
            moe_output = moe_layer(x)
            x = x + moe_output
            
            # 注意力层
            attn_output, _ = attn_layer(x, x, x)
            x = x + attn_output
        
        x = self.norm(x)
        return self.output(x)
```

## 评估指标

### 1. 专家利用率
```python
def calculate_expert_utilization(expert_counts):
    """计算专家利用率"""
    total = expert_counts.sum()
    utilization = expert_counts / total
    
    # 计算熵（越高越好）
    entropy = -torch.sum(utilization * torch.log(utilization + 1e-10))
    max_entropy = torch.log(torch.tensor(len(expert_counts)))
    
    normalized_entropy = entropy / max_entropy
    
    return {
        "utilization": utilization,
        "entropy": entropy.item(),
        "normalized_entropy": normalized_entropy.item()
    }
```

### 2. 路由一致性
```python
def calculate_routing_consistency(routing_decisions):
    """计算路由一致性"""
    # 相似输入应该路由到相同专家
    # 计算路由决策的方差
    routing_variance = torch.var(routing_decisions, dim=0)
    
    return {
        "mean_variance": routing_variance.mean().item(),
        "max_variance": routing_variance.max().item()
    }
```

## 未来方向

### 1. 动态专家
```
概念：根据输入动态调整专家数量
优势：更灵活，更高效
```

### 2. 层级MoE
```
概念：多层MoE架构
优势：更好的专家分工
```

### 3. 跨层专家
```
概念：专家在不同层间共享
优势：参数效率更高
```

## 学习资源
- Mixtral论文
- DeepSeek-V2论文
- MoE综述论文

## 下一步
继续学习 [小模型优化](03_小模型优化.md)