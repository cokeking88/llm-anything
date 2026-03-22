# 多Agent系统

## Agent协作

### 什么是多Agent系统？
**定义**：多个Agent协同工作，完成复杂任务

### 协作模式
```
1. 层级协作：上级Agent分配任务，下级Agent执行
2. 平等协作：Agent之间相互协商
3. 流水线协作：Agent按顺序处理任务
4. 竞争协作：多个Agent竞争解决问题
```

### 实现示例
```python
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI

class MultiAgentSystem:
    def __init__(self, agents):
        self.agents = agents  # Agent字典
        self.shared_memory = {}
    
    def execute_task(self, task, strategy="hierarchical"):
        """执行任务"""
        
        if strategy == "hierarchical":
            return self.hierarchical_execution(task)
        elif strategy == "pipeline":
            return self.pipeline_execution(task)
        elif strategy == "collaborative":
            return self.collaborative_execution(task)
    
    def hierarchical_execution(self, task):
        """层级执行"""
        # 主Agent分析任务
        manager = self.agents["manager"]
        subtasks = manager.analyze_task(task)
        
        results = []
        for subtask in subtasks:
            # 分配给专门的Agent
            specialist = self.agents[subtask["type"]]
            result = specialist.execute(subtask["description"])
            results.append(result)
        
        # 主Agent整合结果
        final_result = manager.integrate_results(results)
        return final_result
```

## 角色分工

### 定义Agent角色
```python
class AgentRole:
    def __init__(self, name, description, capabilities):
        self.name = name
        self.description = description
        self.capabilities = capabilities

# 预定义角色
ROLES = {
    "researcher": AgentRole(
        name="研究员",
        description="负责研究和分析信息",
        capabilities=["搜索", "分析", "总结"]
    ),
    "writer": AgentRole(
        name="写手",
        description="负责内容创作",
        capabilities=["写作", "编辑", "校对"]
    ),
    "coder": AgentRole(
        name="程序员",
        description="负责编程和实现",
        capabilities=["编程", "调试", "测试"]
    ),
    "reviewer": AgentRole(
        name="审核员",
        description="负责质量检查",
        capabilities=["审核", "反馈", "改进"]
    )
}

class RoleBasedAgent:
    def __init__(self, role, llm):
        self.role = role
        self.llm = llm
    
    def execute(self, task):
        """根据角色执行任务"""
        prompt = f"""
        你是一个{self.role.name}。
        {self.role.description}
        
        任务：{task}
        
        请运用你的能力：{', '.join(self.role.capabilities)}
        完成这个任务：
        """
        
        return self.llm.generate(prompt)
```

## 通信机制

### Agent间通信
```python
class AgentCommunication:
    def __init__(self):
        self.message_queue = {}
        self.subscribers = {}
    
    def register_agent(self, agent_id):
        """注册Agent"""
        self.message_queue[agent_id] = []
        self.subscribers[agent_id] = []
    
    def send_message(self, from_agent, to_agent, message):
        """发送消息"""
        message_entry = {
            "from": from_agent,
            "to": to_agent,
            "content": message,
            "timestamp": datetime.now()
        }
        
        self.message_queue[to_agent].append(message_entry)
        
        # 通知订阅者
        for subscriber in self.subscribers.get(to_agent, []):
            subscriber.notify(message_entry)
    
    def get_messages(self, agent_id):
        """获取消息"""
        messages = self.message_queue[agent_id].copy()
        self.message_queue[agent_id].clear()
        return messages
    
    def broadcast(self, from_agent, message):
        """广播消息"""
        for agent_id in self.message_queue.keys():
            if agent_id != from_agent:
                self.send_message(from_agent, agent_id, message)
```

### 消息协议
```python
class MessageProtocol:
    """Agent间通信协议"""
    
    @staticmethod
    def create_request(from_agent, to_agent, action, data):
        """创建请求消息"""
        return {
            "type": "request",
            "from": from_agent,
            "to": to_agent,
            "action": action,
            "data": data,
            "id": str(uuid.uuid4())
        }
    
    @staticmethod
    def create_response(request_id, status, data):
        """创建响应消息"""
        return {
            "type": "response",
            "request_id": request_id,
            "status": status,
            "data": data
        }
    
    @staticmethod
    def create_notification(from_agent, event, data):
        """创建通知消息"""
        return {
            "type": "notification",
            "from": from_agent,
            "event": event,
            "data": data
        }
```

## 任务分配

### 智能任务分配
```python
class TaskAllocator:
    def __init__(self, agents):
        self.agents = agents
        self.agent_capabilities = self._extract_capabilities()
    
    def _extract_capabilities(self):
        """提取Agent能力"""
        capabilities = {}
        for agent_id, agent in self.agents.items():
            capabilities[agent_id] = agent.get_capabilities()
        return capabilities
    
    def allocate_task(self, task):
        """分配任务给最合适的Agent"""
        
        # 分析任务需求
        task_requirements = self._analyze_requirements(task)
        
        # 计算每个Agent的匹配度
        scores = {}
        for agent_id, capabilities in self.agent_capabilities.items():
            score = self._calculate_match_score(
                task_requirements, 
                capabilities
            )
            scores[agent_id] = score
        
        # 选择最佳Agent
        best_agent = max(scores.items(), key=lambda x: x[1])
        
        return best_agent[0], best_agent[1]
    
    def _analyze_requirements(self, task):
        """分析任务需求"""
        # 这里可以使用LLM分析任务需求
        return {"capability": "general", "priority": "normal"}
    
    def _calculate_match_score(self, requirements, capabilities):
        """计算匹配度分数"""
        # 简单的匹配算法
        score = 0
        for req in requirements:
            if req in capabilities:
                score += 1
        return score / len(requirements) if requirements else 0
```

## 协作模式

### 1. 专家委员会模式
```python
class ExpertCommittee:
    """专家委员会模式"""
    
    def __init__(self, experts):
        self.experts = experts  # 各领域专家
    
    def solve_problem(self, problem):
        """解决问题"""
        # 1. 专家独立分析
        expert_opinions = {}
        for expert_name, expert in self.experts.items():
            opinion = expert.analyze(problem)
            expert_opinions[expert_name] = opinion
        
        # 2. 讨论和辩论
        discussion = self._facilitate_discussion(expert_opinions)
        
        # 3. 达成共识
        consensus = self._reach_consensus(discussion)
        
        return consensus
    
    def _facilitate_discussion(self, opinions):
        """促进讨论"""
        # 使用LLM模拟讨论
        prompt = f"""
        以下是各专家的意见：
        {opinions}
        
        请模拟专家讨论，找出共识和分歧：
        """
        return llm.generate(prompt)
    
    def _reach_consensus(self, discussion):
        """达成共识"""
        prompt = f"""
        讨论内容：
        {discussion}
        
        请总结最终共识：
        """
        return llm.generate(prompt)
```

### 2. 流水线模式
```python
class PipelineAgents:
    """流水线模式"""
    
    def __init__(self, stages):
        self.stages = stages  # 各阶段的Agent
    
    def process(self, input_data):
        """处理数据"""
        current_data = input_data
        
        for stage_name, agent in self.stages.items():
            print(f"阶段: {stage_name}")
            current_data = agent.process(current_data)
        
        return current_data
```

## 冲突解决

### 冲突检测和解决
```python
class ConflictResolver:
    def __init__(self, mediator):
        self.mediator = mediator  # 调解Agent
    
    def detect_conflict(self, agent1_view, agent2_view):
        """检测冲突"""
        # 比较两个Agent的观点
        similarity = self._calculate_similarity(
            agent1_view, 
            agent2_view
        )
        
        return similarity < 0.5  # 相似度低于0.5认为是冲突
    
    def resolve_conflict(self, agent1, agent2, conflict_description):
        """解决冲突"""
        # 请调解Agent介入
        resolution = self.mediator.mediate(
            agent1.get_position(),
            agent2.get_position(),
            conflict_description
        )
        
        return resolution
    
    def _calculate_similarity(self, view1, view2):
        """计算相似度"""
        # 使用嵌入计算相似度
        embedding1 = embeddings.embed_query(str(view1))
        embedding2 = embeddings.embed_query(str(view2))
        
        return cosine_similarity(embedding1, embedding2)
```

## 性能监控

### 监控Agent性能
```python
class AgentMonitor:
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, agent_id, metric_name, value):
        """记录指标"""
        if agent_id not in self.metrics:
            self.metrics[agent_id] = {}
        
        if metric_name not in self.metrics[agent_id]:
            self.metrics[agent_id][metric_name] = []
        
        self.metrics[agent_id][metric_name].append({
            "value": value,
            "timestamp": datetime.now()
        })
    
    def get_performance_report(self, agent_id):
        """获取性能报告"""
        if agent_id not in self.metrics:
            return None
        
        report = {}
        for metric_name, values in self.metrics[agent_id].items():
            report[metric_name] = {
                "latest": values[-1]["value"],
                "average": sum(v["value"] for v in values) / len(values),
                "trend": self._calculate_trend(values)
            }
        
        return report
```

## 学习资源
- 多Agent系统论文
- Agent通信协议
- 分布式AI系统

## 下一步
继续学习 [实战项目](06_实战项目.md)