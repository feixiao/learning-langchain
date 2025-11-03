from pydantic import BaseModel
from typing import Optional
import os


# Define a structured output model with answer and justification fields
class AnswerWithJustification(BaseModel):
    """An answer to the user's question along with justification for the answer."""

    answer: str
    """The answer to the user's question"""
    justification: str
    """Justification for the answer"""


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

llm = build_model("ollama")

structured_llm = llm.with_structured_output(AnswerWithJustification)

response = structured_llm.invoke(
    "What weighs more, a pound of bricks or a pound of feathers")
print(response)
