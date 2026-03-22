# 01_LLM基础原理

## 什么是大语言模型？

**一句话**：通过阅读海量文本学会"说话"的AI

**类比**：
```
传统NLP：教小孩做数学题（规则明确）
LLM：教小孩读万卷书后写文章（从数据中学习）

ChatGPT = 读了互联网上几乎所有文字后学会的"超级学生"
```

---

## 核心思想：预测下一个词

**一句话**：LLM的本质就是"预测下一个词"

```
输入：今天天气
模型预测：真
输入：今天天气真
模型预测：好
输入：今天天气真好
模型预测：，
...

逐步生成完整句子
```

**类比**：
```
手机输入法：你打"我"，它预测"们/是/的"
LLM：一样的原理，但更强大
```

---

## Transformer架构

### 什么是Transformer？

**一句话**：基于"注意力机制"的架构，能并行处理序列

**类比**：
```
RNN：像读书，一个字一个字读（串行）
Transformer：像拍照，一眼看到所有字（并行）

RNN：记忆力有限，前面的容易忘
Transformer：每个字都能直接看到所有字
```

### 核心：注意力机制

**直观理解**：注意力就是"关注重点"

```
句子："小明在北京大学学习人工智能"

当处理"学习"时，注意力分配：
- 小明：0.3（主语）
- 北京大学：0.2（地点）
- 学习：0.4（当前词）
- 人工智能：0.1（宾语）
```

**计算公式**：
```
Attention(Q, K, V) = softmax(Q × Kᵀ / √dₖ) × V

Q：Query（我在找什么）
K：Key（我能提供什么）
V：Value（我的内容）
```

### 自注意力图解

```
输入：I love you

每个词都和其他词计算相关性：
        I    love   you
I      1.0    0.3   0.2
love   0.3    1.0   0.8
you    0.2    0.8   1.0

"love"最关注"you"（0.8）
```

---

## Tokenization（分词）

### 什么是Tokenization？

**一句话**：把文字拆成小块（token），让模型能处理

**类比**：
```
中文：一个字一个字
英文：一个词一个词
Token：可能是词、词的一部分、甚至单个字符

"我爱人工智能" → ["我", "爱", "人工", "智能"]
"I love AI" → ["I", "love", " ", "AI"]
```

### BPE算法

**核心思想**：合并出现频率最高的相邻字符对

```
初始：["l", "o", "w", " ", "l", "o", "w", " ", "l", "o", "w", "e", "r"]
合并1：["lo", "w", " ", "lo", "w", " ", "lo", "w", "e", "r"]  (l+o最频繁)
合并2：["low", " ", "low", " ", "low", "e", "r"]  (lo+w)
继续...
```

### 代码示例

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello, world!"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['Hello', ',', ' world', '!']
```

---

## 预训练

### 什么是预训练？

**一句话**：在海量文本上训练模型，让它学会语言规律

**类比**：
```
预训练 = 大学通识教育（学语言、学知识）
微调 = 职业培训（学专业技能）
```

### 预训练任务

**语言建模（LM）**：
```
输入：The cat sat on the
目标：mat

模型学习：P("mat" | "The cat sat on the")
```

**掩码语言建模（MLM，BERT用）**：
```
输入：The [MASK] sat on the mat
目标：cat
```

### 预训练数据

| 数据集 | 内容 | 大小 |
|--------|------|------|
| Common Crawl | 网页 | PB级 |
| Wikipedia | 维基百科 | 20GB+ |
| Books | 书籍 | 100GB+ |
| GitHub | 代码 | TB级 |

---

## 模型架构对比

### Encoder-only（BERT类）

```
双向理解 → 适合理解任务
├── 文本分类
├── 命名实体识别
└── 语义相似度
```

### Decoder-only（GPT类）

```
单向生成 → 适合生成任务
├── 文本生成
├── 对话
└── 代码生成
```

### Encoder-Decoder（T5类）

```
编码+解码 → 适合序列转换
├── 机器翻译
├── 文本摘要
└── 问答
```

---

## 推理优化

### 量化（Quantization）

**核心思想**：用更少的位数表示参数

```
FP32（32位）→ FP16（16位）→ INT8（8位）→ INT4（4位）

精度降低，但模型变小、速度变快
```

**效果**：
```
7B模型：
- FP16：14GB显存
- INT4：4GB显存（省70%）
```

### KV Cache

**核心思想**：缓存之前的计算结果

```
传统：每次生成都重新计算所有token
KV Cache：只计算新token，复用之前的

速度提升：10倍以上
```

---

## 常见问题

### Q1：LLM为什么会产生幻觉？

**原因**：模型只学会了"统计规律"，不理解"真假"

```
训练数据中有："地球是平的"（错误信息）
模型学到："地球"后面可能接"是平的"

模型不知道什么是真的，只知道什么"看起来像真的"
```

### Q2：如何选择LLM？

| 需求 | 推荐模型 |
|------|----------|
| 最强能力 | GPT-4o、Claude 3.5 |
| 开源可部署 | LLaMA 3、Qwen 2 |
| 中文任务 | 通义千问、文心一言 |
| 代码任务 | DeepSeek-Coder |
| 低成本 | 小模型 + 量化 |

---

## 学习资源
- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Hugging Face课程](https://huggingface.co/learn)

## 下一步
学习 [[02_Prompt工程]]
