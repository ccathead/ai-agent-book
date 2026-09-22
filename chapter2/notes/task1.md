# chapter 2

## 1. 正文

上下文组成部分包括哪些方面？

Agent 是使用API消息结构来定义上下文工程的

大模型 API 的核心是一个**消息列表**（messages），列表中的每条消息都有一个**角色**（role）标识，模型根据角色来理解每条消息的含义和来源：

system:系统提示词

user：用户消息

assistant：模型回复

tool：工具执行结果

对应到代码里面：

```JSON
# —— tool definitions ——
# ── Initial message list ──
```

要设计对KV Cache 友好的上下文设计，如何设计？

如何 提高 KV Cache 命中

## 2.实验

### 实验2-1

### 实验2-3
