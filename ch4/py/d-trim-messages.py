from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    trim_messages,
)

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


# Define sample messages
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]


model = build_model("ollama")
# Create trimmer
# 触发了基于 tokenizer 的“token 计数”
# 你在 d-trim-messages.py 里这样写：trim_messages(..., token_counter=model, ...)。
# 对非 OpenAI 模型，LangChain 可能会调用 Transformers 的分词器来做 token 计数，从而需要下载相应 tokenizer（来源就是 huggingface.co）。
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Apply trimming
trimmed = trimmer.invoke(messages)
print(trimmed)
