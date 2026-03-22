# Tokenization分词技术

## 什么是Tokenization？

### 一句话定义
**一句话**：把文字拆成小块（token），让模型能处理

### 类比理解
```
中文：一个字一个字
英文：一个词一个词
Token：可能是词、词的一部分、甚至单个字符

"我爱人工智能" → ["我", "爱", "人工", "智能"]
"I love AI" → ["I", "love", " ", "AI"]
```

## 为什么需要Tokenization？

### 1. 模型只能处理数字
```
问题：神经网络只能处理数值数据
解决：将文本映射为数字序列
过程：文本 → Token → 数字ID
```

### 2. 处理未登录词（OOV）
```
问题：如何处理训练时没见过的词
解决：子词分割，将未知词拆分为已知子词
示例：
  "unhappiness" → ["un", "happy", "ness"]
  "chatgpt" → ["chat", "g", "pt"]
```

### 3. 平衡词汇表大小
```
问题：词汇表太大 → 参数多，内存大
问题：词汇表太小 → OOV问题严重
解决：子词分割平衡两者
```

## Tokenization方法对比

### 1. 词级分词（Word-level）
```
方法：按空格和标点分割
优点：直观，易于理解
缺点：词汇表巨大，OOV问题严重
示例：
  "I love AI" → ["I", "love", "AI"]
```

### 2. 字符级分词（Character-level）
```
方法：每个字符单独分割
优点：词汇表小，无OOV问题
缺点：序列太长，语义信息丢失
示例：
  "I love AI" → ["I", " ", "l", "o", "v", "e", " ", "A", "I"]
```

### 3. 子词分词（Subword-level）
```
方法：平衡词和字符的优点
优点：词汇表适中，处理OOV
缺点：需要训练分词器
示例：
  "unhappiness" → ["un", "happy", "ness"]
  "I love AI" → ["I", "love", "AI"]
```

## 主流分词算法

### 1. BPE（Byte Pair Encoding）

#### 核心思想
```
合并出现频率最高的相邻字符对
迭代直到达到目标词汇表大小
```

#### BPE算法步骤
```
步骤1：初始化词汇表（所有字符）
步骤2：统计所有相邻对的频率
步骤3：合并频率最高的对
步骤4：重复步骤2-3直到满足条件
```

#### BPE示例
```
初始词汇表：{"l", "o", "w", "e", "r", " ", "s", "t", "i", "d", "n"}

初始分割：
  "low lower" → ["l", "o", "w", " ", "l", "o", "w", "e", "r"]

迭代1：合并"l"+"o"="lo"（出现2次）
  → ["lo", "w", " ", "lo", "w", "e", "r"]

迭代2：合并"lo"+"w"="low"（出现2次）
  → ["low", " ", "low", "e", "r"]

迭代3：合并"e"+"r"="er"（出现1次）
  → ["low", " ", "low", "er"]

最终：["low", " ", "low", "er"]
```

### 2. WordPiece

#### 核心思想
```
类似BPE，但合并标准不同
选择最大化似然的合并
Google BERT使用
```

#### 合并标准
```
选择使训练数据似然最大的合并
最大化 P(合并后的词) / (P(词1) × P(词2))
```

### 3. Unigram Language Model

#### 核心思想
```
从大词汇表开始，逐步删除
使用语言模型损失函数
SentencePiece常用
```

#### 算法步骤
```
步骤1：初始化大词汇表（例如68万）
步骤2：计算每个子词的损失
步骤3：删除损失最小的子词（例如20%）
步骤4：重复步骤2-3直到达到目标大小
```

### 4. SentencePiece

#### 特点
```
语言无关：不需要预分词
支持BPE和Unigram
直接处理原始文本
```

#### 示例
```
原始文本："我爱人工智能"
SentencePiece → ["▁我", "爱", "人工", "智能"]
注意："▁"表示空格
```

## 实际应用示例

### GPT系列分词器
```
算法：BPE
词汇表：50,257个token
特点：支持代码和自然语言
```

### BERT分词器
```
算法：WordPiece
词汇表：30,000个token
特点：支持103种语言
```

### LLaMA分词器
```
算法：BPE
词汇表：32,000个token
特点：支持多语言，代码友好
```

### Qwen分词器
```
算法：BPE
词汇表：151,936个token
特点：中文优化，支持代码
```

## 代码示例

### Hugging Face Tokenizers
```python
from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 分词示例
text = "Hello, world!"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['Hello', ',', ' world', '!']

# 编码为ID
ids = tokenizer.encode(text)
print(ids)  # [15496, 11, 995, 0]

# 解码回文本
decoded = tokenizer.decode(ids)
print(decoded)  # "Hello, world!"
```

### SentencePiece示例
```python
import sentencepiece as spm

# 训练分词器
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='m',
    vocab_size=1000
)

# 加载分词器
sp = spm.SentencePieceProcessor()
sp.load('m.model')

# 分词
text = "我爱人工智能"
tokens = sp.encode_as_pieces(text)
print(tokens)  # ['▁我', '爱', '人工', '智能']

# 编码
ids = sp.encode_as_ids(text)
print(ids)  # [100, 200, 300, 400]
```

## 分词器评估指标

### 1. 词汇表大小
```
影响：模型参数数量
权衡：大小 vs 覆盖率
```

### 2. 编码效率
```
定义：平均每个token代表的字符数
计算：总字符数 / 总token数
目标：越高越好
```

### 3. OOV率
```
定义：未登录词的比例
影响：模型处理新词的能力
目标：越低越好
```

### 4. 解码一致性
```
定义：编码后能否完全解码回原文
重要性：确保信息无损
```

## 分词最佳实践

### 1. 选择适合的分词器
```
通用文本：BPE或WordPiece
多语言：SentencePiece
代码：考虑代码友好的分词器
```

### 2. 调整词汇表大小
```
小模型：较小词汇表（如32K）
大模型：较大词汇表（如100K+）
多语言：需要更大词汇表
```

### 3. 处理特殊token
```
常见特殊token：
  <unk>：未知词
  <pad>：填充
  <s>：句子开始
  </s>：句子结束
  <bos>：序列开始
  <eos>：序列结束
```

## 分词对模型性能的影响

### 1. 训练效率
```
更长的序列：训练更慢
更短的序列：训练更快
建议：平衡序列长度和信息保留
```

### 2. 模型效果
```
更好的分词：更准确的语义表示
更差的分词：语义信息丢失
建议：为领域定制分词器
```

### 3. 推理速度
```
更多token：推理更慢
更少token：推理更快
建议：优化分词以提高效率
```

## 学习资源
- Hugging Face Tokenizers文档
- SentencePiece论文
- BPE算法原始论文

## 下一步
继续学习 [预训练原理](04_预训练.md)