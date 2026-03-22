# Agent基础

## 什么是AI Agent？

### 一句话定义
**一句话**：AI Agent = 能思考、能行动的AI助手

### 类比理解
```
普通LLM：只能聊天，像一个"只会说话的顾问"
AI Agent：能使用工具，像一个"能干活的助手"

普通LLM："我帮你查一下天气"（但其实查不了）
AI Agent：*调用天气API* "今天北京晴天，25度"
```

## Agent核心能力

### 1. 思考能力
```
功能：理解任务，制定计划
实现：推理、分析、决策
示例：分析复杂问题，制定解决方案
```

### 2. 行动能力
```
功能：使用工具完成任务
实现：工具调用、API调用、代码执行
示例：调用天气API、执行Python代码
```

### 3. 记忆能力
```
功能：记住上下文和历史
实现：短期记忆、长期记忆
示例：记住对话历史、用户偏好
```

### 4. 反思能力
```
功能：检查结果，调整策略
实现：自我评估、错误纠正
示例：检查答案是否正确，调整方法
```

## Agent架构

### 核心架构
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

### 工作流程
```
1. 接收用户输入
2. 思考：理解任务，制定计划
3. 行动：调用工具执行任务
4. 观察：获取工具执行结果
5. 反思：评估结果，决定下一步
6. 输出：返回最终结果
```

## ReAct框架

### 什么是ReAct？
**核心思想**：推理（Reasoning）+ 行动（Acting）

### 工作流程
```
问题：北京今天天气怎么样？

思考：我需要查询北京的天气信息
行动：调用天气API，参数：城市=北京
观察：北京今天晴天，25度
思考：我已经获得了天气信息
答案：北京今天晴天，温度25度
```

### ReAct实现
```python
from langchain.agents import initialize_agent, Tool

# 定义工具
tools = [
    Tool(
        name="Search",
        func=search_function,
        description="搜索互联网获取信息"
    ),
    Tool(
        name="Calculator",
        func=calculator_function,
        description="计算数学表达式"
    )
]

# 初始化Agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# 运行Agent
result = agent.run("北京今天天气怎么样？")
```

## 工具调用

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

### 工具调用原理
```
1. LLM决定调用哪个工具
2. LLM生成工具参数
3. 执行工具
4. 返回结果给LLM
5. LLM基于结果继续推理
```

### 工具集成
```python
from langchain.agents import AgentExecutor

# 创建工具列表
tools = [calculator, search_web, get_weather]

# 创建Agent
agent = create_react_agent(llm, tools, prompt)

# 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# 执行
result = agent_executor.invoke({
    "input": "计算 15 * 23 + 45"
})
```

## 记忆系统

### 短期记忆
```
功能：记住当前对话上下文
实现：对话历史缓存
示例：记住用户刚说的内容
```

### 长期记忆
```
功能：记住重要信息
实现：向量数据库、知识图谱
示例：记住用户偏好、历史对话
```

### 记忆管理
```python
from langchain.memory import ConversationBufferMemory

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 使用记忆
agent = initialize_agent(
    tools,
    llm,
    agent="conversational-react-description",
    memory=memory,
    verbose=True
)
```

## 规划能力

### 任务分解
```
复杂任务："写一篇关于AI的文章"
分解为：
  1. 研究AI最新进展
  2. 设计文章结构
  3. 撰写引言
  4. 撰写主体
  5. 撰写结论
  6. 编辑和校对
```

### 思维链规划
```python
# 规划提示
planning_prompt = """
请为以下任务制定详细计划：

任务：{task}

要求：
1. 分解为子任务
2. 确定子任务顺序
3. 估计每个子任务所需时间
4. 确定所需资源

输出格式：
子任务1：{描述}，预计时间：{时间}
子任务2：{描述}，预计时间：{时间}
...
"""
```

## 实战示例

### 客服助手Agent
```python
# 工具定义
tools = [
    Tool(name="查订单", func=query_order, description="查询订单信息"),
    Tool(name="退换货", func=return_item, description="处理退换货"),
    Tool(name="转人工", func=transfer_to_human, description="转接人工客服")
]

# 系统提示
system_prompt = """
你是一个专业的客服助手。你的职责是：
1. 礼貌地回应客户
2. 帮助客户解决问题
3. 必要时转接人工客服

你不能：
1. 做出无法兑现的承诺
2. 提供错误信息
3. 与客户争吵
"""

# 创建Agent
agent = create_customer_service_agent(llm, tools, system_prompt)
```

### 代码助手Agent
```python
# 工具定义
tools = [
    Tool(name="运行代码", func=run_code, description="执行Python代码"),
    Tool(name="搜索文档", func=search_docs, description="搜索技术文档"),
    Tool(name="代码审查", func=code_review, description="审查代码质量")
]

# 系统提示
system_prompt = """
你是一个专业的编程助手。你的职责是：
1. 帮助用户编写代码
2. 解释代码功能
3. 调试和修复错误
4. 优化代码性能

你总是：
1. 提供清晰的代码示例
2. 解释代码原理
3. 考虑最佳实践
"""
```

## 性能优化

### 1. 工具优化
```
原则：提供准确的工具描述
做法：明确参数类型、返回值
```

### 2. 提示优化
```
原则：清晰的系统提示
做法：定义角色、职责、限制
```

### 3. 记忆优化
```
原则：平衡记忆长度和效果
做法：只记住关键信息
```

## 学习资源
- LangChain Agent文档
- ReAct论文
- Agent开发教程

## 下一步
继续学习 [工具调用](02_工具调用.md)