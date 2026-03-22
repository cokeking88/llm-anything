# 05_Agent开发

## 什么是AI Agent？

**一句话**：AI Agent = 能思考、能行动的AI助手

**类比**：
```
普通LLM：只能聊天，像一个"只会说话的顾问"
AI Agent：能使用工具，像一个"能干活的助手"

普通LLM："我帮你查一下天气"（但其实查不了）
AI Agent：*调用天气API* "今天北京晴天，25度"
```

---

## Agent核心能力

1. **思考**：理解任务，制定计划
2. **行动**：使用工具完成任务
3. **记忆**：记住上下文和历史
4. **反思**：检查结果，调整策略

---

## Agent架构

```
┌─────────────────────────────────────┐
│              Agent                   │
│  ┌─────────────────────────────┐   │
│  │           LLM大脑           │   │
│  └─────────────────────────────┘   │
│                  ↓                  │
│  ┌─────────┬─────────┬─────────┐   │
│  │  工具1   │  工具2   │  工具3   │   │
│  │ (搜索)   │ (计算)   │ (代码)   │   │
│  └─────────┴─────────┴─────────┘   │
│                  ↓                  │
│  ┌─────────────────────────────┐   │
│  │          记忆模块            │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

---

## ReAct框架

**核心思想**：推理（Reasoning）+ 行动（Acting）

```
问题：北京今天天气怎么样？

思考：我需要查询北京的天气信息
行动：调用天气API，参数：城市=北京
观察：北京今天晴天，25度
思考：我已经获得了天气信息
答案：北京今天晴天，温度25度
```

---

## 工具调用

### 什么是工具调用？

**一句话**：让LLM能够使用外部工具（API、函数等）

**类比**：
```
人：遇到数学题，用计算器算
Agent：遇到数学题，调用计算工具
```

### 工具定义

```python
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

@tool
def search_web(query: str) -> str:
    """搜索互联网获取信息"""
    return f"搜索结果：关于{query}的信息..."

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}今天晴天，25度"
```

### 创建Agent

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# 创建LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 工具列表
tools = [calculator, search_web, get_weather]

# 创建Agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 运行
result = agent_executor.invoke({"input": "北京天气怎么样？"})
```

---

## 记忆机制

### 短期记忆

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

# 保存对话
memory.save_context({"input": "我叫小明"}, {"output": "好的，记住了"})

# 加载记忆
memory.load_memory_variables({})
```

### 长期记忆

```python
from langchain.memory import VectorStoreRetrieverMemory

# 向量存储记忆
memory = VectorStoreRetrieverMemory(retriever=retriever)
memory.save_context({"input": "我喜欢吃火锅"}, {"output": "记住了"})
```

---

## 规划能力

### Plan-and-Execute

```python
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner
)

# 创建规划器
planner = load_chat_planner(llm)

# 创建执行器
executor = load_agent_executor(llm, tools)

# 创建Agent
agent = PlanAndExecute(planner=planner, executor=executor)

# 运行
result = agent.run("分析一下AI行业发展趋势")
```

---

## 多Agent系统

### 什么是多Agent？

**一句话**：多个Agent协作完成复杂任务

**类比**：
```
单Agent：一个人完成所有工作
多Agent：一个团队分工合作

- 研究Agent：负责搜索信息
- 写作Agent：负责撰写内容
- 审核Agent：负责检查质量
```

### 实现

```python
# 研究Agent
research_agent = AgentExecutor(
    agent=create_openai_functions_agent(llm, [search_web]),
    tools=[search_web]
)

# 写作Agent
writing_agent = AgentExecutor(
    agent=create_openai_functions_agent(llm, [calculator]),
    tools=[calculator]
)

# 调度
def coordinate(task):
    research = research_agent.invoke({"input": f"研究：{task}"})
    writing = writing_agent.invoke({"input": f"写作：{research['output']}"})
    return writing["output"]
```

---

## 完整项目：智能助手

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory

# 定义工具
@tool
def search(query: str) -> str:
    """搜索互联网"""
    return f"搜索结果：{query}的最新信息"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    """获取天气"""
    return f"{city}今天晴天，25度"

# 创建Agent
llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [search, calculate, get_weather]
memory = ConversationBufferMemory(memory_key="chat_history")

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

# 使用
print(agent_executor.invoke({"input": "你好，我叫小明"})["output"])
print(agent_executor.invoke({"input": "北京天气怎么样？"})["output"])
print(agent_executor.invoke({"input": "帮我算 15*23+47"})["output"])
print(agent_executor.invoke({"input": "我叫什么名字？"})["output"])  # 会记住
```

---

## Agent框架对比

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| LangChain | 功能全面、文档丰富 | 通用开发 |
| AutoGPT | 自主执行复杂任务 | 实验性项目 |
| CrewAI | 多Agent协作 | 团队协作 |
| MetaGPT | 软件开发Agent | 代码生成 |

---

## 学习资源
- [LangChain Agent文档](https://docs.langchain.com/docs/modules/agents)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)

## 下一步
学习 [[06_多模态LLM]] 和 [[07_前沿技术]]
