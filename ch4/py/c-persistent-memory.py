from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Optional
import os


def build_model(provider: str, model_name: Optional[str] = None):
	provider = provider.lower()

	if provider == "openai":
		# 延迟导入，避免未安装依赖或本地无用时报错
		from langchain_openai import ChatOpenAI

		name = model_name or os.getenv("LLM_MODEL") or "gpt-4o-mini"
		return ChatOpenAI(model=name)

	if provider == "ollama":
		# 延迟导入，避免未安装依赖或本地无用时报错
		from langchain_ollama import ChatOllama

		name = model_name or os.getenv("LLM_MODEL") or "deepseek-r1:14b"
		return ChatOllama(model=name)

	raise ValueError(
		f"Unsupported provider: {provider}. Use 'openai' or 'ollama'."
	)



model = build_model("ollama")

class State(TypedDict):
    messages: Annotated[list, add_messages]


builder = StateGraph(State)



def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Add persistence with MemorySaver
graph = builder.compile(checkpointer=MemorySaver())

# Configure thread
thread1 = {"configurable": {"thread_id": "1"}}

# Run with persistence
result_1 = graph.invoke({"messages": [HumanMessage("hi, my name is Jack!")]}, thread1)
print(result_1)

result_2 = graph.invoke({"messages": [HumanMessage("what is my name?")]}, thread1)
print(result_2)

# Get state
print(graph.get_state(thread1))
