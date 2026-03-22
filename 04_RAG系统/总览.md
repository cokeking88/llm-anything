# 04_RAG系统

## 什么是RAG？

**一句话**：RAG = 先检索相关文档，再让LLM根据文档生成答案

**类比**：
```
普通LLM：开卷考试，但只能凭记忆答题
RAG：开卷考试，可以翻书找答案

RAG让LLM能"查资料"再回答
```

---

## RAG架构

```
用户问题
    ↓
┌─────────────────┐
│    检索器        │  ← 从知识库中找到相关文档
└────────┬────────┘
         ↓
    相关文档片段
         ↓
┌─────────────────┐
│    生成器        │  ← LLM根据文档生成答案
└────────┬────────┘
         ↓
      最终答案
```

---

## RAG vs 微调

| 特性 | RAG | 微调 |
|------|-----|------|
| 知识更新 | 实时更新 | 需要重新训练 |
| 幻觉风险 | 低（有依据） | 中等 |
| 实现难度 | 中等 | 较高 |
| 成本 | 较低 | 较高 |
| 适用场景 | 知识问答 | 风格/能力调整 |

---

## 文档处理

### 文档加载

```python
from langchain.document_loaders import PyPDFLoader, TextLoader

# 加载PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 加载文本
loader = TextLoader("document.txt")
documents = loader.load()
```

### 文档分块

**为什么分块？**
- Embedding模型有长度限制
- 小块更精确匹配
- 减少噪声

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块500字符
    chunk_overlap=50     # 重叠50字符
)

chunks = text_splitter.split_documents(documents)
```

**分块大小选择**：
| 文档类型 | 推荐块大小 | 重叠大小 |
|----------|------------|----------|
| 一般文档 | 500-1000 | 100-200 |
| 代码 | 300-500 | 50-100 |

---

## 向量数据库

### 什么是向量数据库？

**一句话**：存储和检索向量的数据库

**类比**：
```
传统数据库：按关键词搜索（精确匹配）
向量数据库：按含义搜索（语义匹配）

搜索"苹果"：
传统：只返回包含"苹果"的结果
向量：还返回"水果"、"iPhone"等相关结果
```

### Embedding模型

**什么是Embedding？**
```
文字 → 向量（数字数组）

"你好" → [0.2, -0.5, 0.8, ...]  （768维）
"Hello" → [0.3, -0.4, 0.7, ...]  （相似的向量）
```

**常用模型**：
| 模型 | 维度 | 特点 |
|------|------|------|
| text-embedding-ada-002 | 1536 | OpenAI，效果好 |
| BGE | 768 | 中文优秀 |
| all-MiniLM-L6-v2 | 384 | 轻量快速 |

### 向量数据库选择

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| FAISS | 本地、快速 | 小规模、实验 |
| Chroma | 简单易用 | 原型开发 |
| Pinecone | 云服务 | 生产环境 |
| Milvus | 开源、可扩展 | 大规模部署 |

---

## 构建RAG系统

### 完整流程

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# 1. 创建Embedding
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")

# 2. 创建向量数据库
vectorstore = FAISS.from_documents(chunks, embeddings)

# 3. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. 创建RAG链
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    retriever=retriever
)

# 5. 查询
result = qa_chain.run("什么是机器学习？")
print(result)
```

### 保存和加载

```python
# 保存
vectorstore.save_local("faiss_index")

# 加载
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

---

## 高级技巧

### 混合检索

```python
# 结合关键词和语义检索
from langchain.retrievers import EnsembleRetriever, BM25Retriever

# BM25（关键词）
bm25_retriever = BM25Retriever.from_documents(chunks)

# 向量检索
vector_retriever = vectorstore.as_retriever()

# 混合
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)
```

### 对话式RAG

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(memory_key="chat_history")

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# 多轮对话
result1 = qa_chain({"question": "什么是机器学习？"})
result2 = qa_chain({"question": "它和深度学习有什么区别？"})
```

---

## 完整项目：知识库问答

```python
class KnowledgeQA:
    def __init__(self, docs_path):
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
        self.llm = OpenAI(temperature=0)
        
        # 加载文档
        loader = DirectoryLoader(docs_path)
        documents = loader.load()
        
        # 分块
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        chunks = text_splitter.split_documents(documents)
        
        # 创建向量库
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # 创建QA链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever()
        )
    
    def query(self, question):
        return self.qa_chain.run(question)

# 使用
qa = KnowledgeQA("./documents")
answer = qa.query("什么是机器学习？")
```

---

## 学习资源
- [LangChain文档](https://docs.langchain.com/)
- [FAISS文档](https://faiss.ai/)

## 下一步
学习 [[05_Agent开发]]
