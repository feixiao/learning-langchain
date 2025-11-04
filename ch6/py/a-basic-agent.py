import ast
from typing import Annotated, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 这段代码就是用 LangGraph 把“聊天模型 + 工具（搜索、计算器）”编织成一个小型 Agent：
# 模型先读消息，判断是否调用工具；如果要，就执行工具并把结果塞回消息，再让模型产出更准确的回答。
# 通过图的条件边把“模型→工具→模型”的循环串起来，从而实现基础的工具增强型智能体。

@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query) # 使用 ast.literal_eval 评估传入的数学表达式字符串，如 "1 + 2*3"。


# 一个网页搜索工具（DuckDuckGo 搜索）
search = DuckDuckGoSearchRun()

# 把两个工具注册。
tools = [search, calculator]
# bind_tools 让模型具备“函数调用/工具调用”的能力
model = ChatOpenAI(temperature=0.1).bind_tools(tools)

# 状态（State）
# 定义为一个 TypedDict，只有一个键 messages，类型是 list，并用 add_messages 这个 reducer 来追加消息。
#在 LangGraph 的状态机里，messages 会不断累积模型与工具的消息。
class State(TypedDict):
    messages: Annotated[list, add_messages]


def model_node(state: State) -> State:
    res = model.invoke(state["messages"])
    return {"messages": res}

# 构建图:
# 添加 "model" 节点和 "tools" 节点；
# 设置从 START 到 "model" 的边；
# 设置从 "model" 条件跳转到 "tools" 或结束（由 tools_condition 判断）；
# 设置从 "tools" 回到 "model" 的边，形成闭环。
builder = StateGraph(State)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
# tools_condition 会检查模型输出是否包含“工具调用指令”，如果有，就把流程导向 ToolNode 执行
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")


# 把图结构变成可执行的 graph
graph = builder.compile()

# Example usage
# "How old was the 30th president of the United States when he died?"
# 模型可能会调用搜索工具查资料（第30任美国总统：Calvin Coolidge；去世时的年龄）。
# 得到搜索结果后，再由模型整合出答案。
input = {
    "messages": [
        HumanMessage(
            "How old was the 30th president of the United States when he died?"
        )
    ]
}

for c in graph.stream(input):
    print(c)
