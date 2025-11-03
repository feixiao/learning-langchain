"""
支持 OpenAI 与 Ollama 的最小示例。

用法：
  python ch1/py/a-llm.py --provider ollama --model deepseek-r1:14b --prompt "The sky is"
  python ch1/py/a-llm.py --provider openai --model gpt-4o-mini --prompt "The sky is"

也可以通过环境变量控制：
  LLM_PROVIDER=ollama|openai
  LLM_MODEL=<model name>
  （OpenAI 需要设置 OPENAI_API_KEY）
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from dotenv import load_dotenv


def build_model(provider: str, model_name: Optional[str] = None):
	provider = provider.lower()

	if provider == "openai":
		# 延迟导入，避免未安装依赖或本地无用时报错
		from langchain_openai import ChatOpenAI

		name = model_name or os.getenv("LLM_MODEL") or "gpt-4o-mini"
		return ChatOpenAI(model=name)

	if provider == "ollama":
		# 延迟导入，避免未安装依赖或本地无用时报错
		from langchain_community.chat_models import ChatOllama

		name = model_name or os.getenv("LLM_MODEL") or "deepseek-r1:14b"
		return ChatOllama(model=name)

	raise ValueError(
		f"Unsupported provider: {provider}. Use 'openai' or 'ollama'."
	)


def main():
	load_dotenv()

	parser = argparse.ArgumentParser(description="Chat demo with OpenAI or Ollama")
	parser.add_argument(
		"--provider",
		choices=["openai", "ollama"],
		default=os.getenv("LLM_PROVIDER", "ollama"),
		help="LLM provider to use",
	)
	parser.add_argument(
		"--model",
		default=os.getenv("LLM_MODEL"),
		help="Model name to use (overrides env)",
	)
	parser.add_argument(
		"--prompt",
		default="The sky is",
		help="Prompt text",
	)
	args = parser.parse_args()

	model = build_model(args.provider, args.model)
	result = model.invoke(args.prompt)

	# AIMessage 有 .content；个别实现也可能直接返回 str
	content = getattr(result, "content", result)
	print(content)


if __name__ == "__main__":
	main()
