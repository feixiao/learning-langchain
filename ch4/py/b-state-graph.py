from typing import Annotated, TypedDict
from typing import Optional
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

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

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    ("placeholder", "{messages}"),
])

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function in the annotation defines how this state should
    # be updated (in this case, it appends new messages to the
    # list, rather than replacing the previous messages)
    messages: Annotated[list, add_messages]


builder = StateGraph(State)

model = build_model("ollama")


def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


# Add the chatbot node. Nodes represent units of work.
builder.add_node("chatbot", chatbot)

# Add edges to define the flow of the graph.
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# Run the graph
input = {"messages": [HumanMessage("hi!")]}
for chunk in graph.stream(input):
    print(chunk)
