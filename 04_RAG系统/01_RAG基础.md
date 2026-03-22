# RAG基础

## 什么是RAG？

### 一句话定义
**一句话**：RAG = 先检索相关文档，再让LLM根据文档生成答案

### 类比理解
```
普通LLM：开卷考试，但只能凭记忆答题
RAG：开卷考试，可以翻书找答案

RAG让LLM能"查资料"再回答
```

## RAG vs 微调

### RAG特点
```
知识更新：实时更新
幻觉风险：低（有依据）
实现难度：中等
成本：较低
适用场景：知识问答
```

### 微调特点
```
知识更新：需要重新训练
幻觉风险：中等
实现难度：较高
成本：较高
适用场景：风格/能力调整
```

### 选择建议
```
选择RAG：
  - 知识频繁更新
  - 需要引用来源
  - 资源有限
  - 需要解释性

选择微调：
  - 风格调整
  - 能力提升
  - 资源充足
  - 性能优先
```

## RAG架构

### 核心组件
```
1. 文档处理：加载、分块、嵌入
2. 向量数据库：存储和检索向量
3. 检索器：找到相关文档
4. 生成器：LLM生成答案
```

### 工作流程
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

### 详细流程
```
步骤1：文档预处理
  - 加载文档（PDF、TXT、HTML等）
  - 文档分块（Chunking）
  - 生成嵌入（Embedding）
  - 存储到向量数据库

步骤2：查询处理
  - 用户输入问题
  - 问题嵌入
  - 向量相似度搜索
  - 返回相关文档

步骤3：答案生成
  - 构建提示：问题 + 相关文档
  - LLM生成答案
  - 返回结果
```

## RAG优势

### 1. 实时知识更新
```
优势：无需重新训练模型
实现：更新知识库即可
示例：新产品信息、最新政策
```

### 2. 减少幻觉
```
优势：基于文档回答，减少编造
实现：要求模型基于文档回答
示例：提供引用来源
```

### 3. 可解释性
```
优势：可以查看引用来源
实现：返回相关文档片段
示例：答案附带引用
```

### 4. 成本效益
```
优势：比微调成本低
实现：无需训练，只需检索
示例：小型团队也能构建
```

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
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块500字符
    chunk_overlap=50     # 重叠50字符
)

chunks = text_splitter.split_documents(documents)
```

### 文本嵌入
```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query("测试文本")
```

## 向量数据库

### 什么是向量数据库？
**一句话**：存储和检索向量的数据库

### 主流向量数据库
```
1. Chroma：轻量级，易于使用
2. FAISS：Facebook开源，高性能
3. Pinecone：托管服务，易扩展
4. Weaviate：开源，功能丰富
5. Milvus：开源，大规模
```

### 向量搜索
```python
import chroma

# 创建数据库
chroma_client = chroma.Client()
collection = chroma_client.create_collection("documents")

# 添加文档
collection.add(
    documents=["文档1", "文档2"],
    ids=["1", "2"]
)

# 搜索
results = collection.query(
    query_texts=["查询问题"],
    n_results=3
)
```

## 检索技术

### 检索器类型
```
1. 向量检索：基于嵌入相似度
2. 关键词检索：基于BM25
3. 混合检索：结合两者
```

### 混合检索
```python
from langchain.retrievers import EnsembleRetriever

# 创建两个检索器
vector_retriever = vectorstore.as_retriever()
bm25_retriever = BM25Retriever.from_documents(documents)

# 组合
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)
```

## 生成优化

### 提示工程
```
RAG提示模板：
"基于以下上下文回答问题：
上下文：{context}
问题：{question}
请基于上下文回答，如果上下文没有相关信息，请说明。"
```

### 上下文压缩
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 创建压缩器
compressor = LLMChainExtractor.from_llm(llm)

# 创建压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

## 实战项目

### 知识库问答系统
```python
from langchain.chains import RetrievalQA

# 创建QA链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 查询
result = qa_chain({"query": "什么是RAG？"})
print(result["result"])
```

### 构建完整RAG系统
```python
# 1. 加载文档
documents = load_documents("docs/")

# 2. 分块
chunks = split_documents(documents)

# 3. 嵌入和存储
vectorstore = create_vectorstore(chunks)

# 4. 创建检索器
retriever = vectorstore.as_retriever()

# 5. 创建QA链
qa_chain = create_qa_chain(llm, retriever)

# 6. 使用
answer = qa_chain.run("用户问题")
```

## 性能优化

### 1. 分块优化
```
原则：平衡信息密度和检索精度
建议：500-1000字符，50-100字符重叠
```

### 2. 检索优化
```
方法：混合检索、重排序
参数：调整top_k值
```

### 3. 提示优化
```
原则：清晰的指令，足够的上下文
技巧：要求引用来源
```

## 学习资源
- LangChain RAG教程
- RAG论文：Retrieval-Augmented Generation
- 向量数据库文档

## 下一步
继续学习 [文档处理](02_文档处理.md)