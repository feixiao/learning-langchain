from langchain_core.messages import HumanMessage, SystemMessage

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

# SystemMessage
# 表示“系统级指令”（role=system）。
# 用来给模型设定全局的行为规范、身份、语气、输出格式、边界等。
# 作用是“约束/指导”后续对话的回答方式，通常优先级高于普通用户消息。
# 典型用途：要求模型只用中文回答、固定返回 JSON、扮演某种角色（如审校员）、遵守安全策略等。

# A message setting the instructions the AI should follow, with the system role
system_msg = SystemMessage(
    "You are a helpful assistant that responds to questions with three exclamation marks."
)


# HumanMessage
# 表示“用户输入”（role=user）。
# 是模型要回答的具体问题或任务内容。
# 典型用途：用户提问、给出任务、提供上下文。
# A message sent from the perspective of the human, with the user role
human_msg = HumanMessage("What is the capital of France?")

response = model.invoke([system_msg, human_msg])
print(response.content)
