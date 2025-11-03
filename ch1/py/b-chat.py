"""
支持 OpenAI 与 Ollama 的聊天示例。

用法：
  python ch1/py/b-chat.py --provider ollama --model deepseek-r1:14b --prompt "What is the capital of France?"
  python ch1/py/b-chat.py --provider openai --model gpt-4o-mini --prompt "What is the capital of France?"

环境变量也可控制：
  LLM_PROVIDER=ollama|openai
  LLM_MODEL=<model name>
  （OpenAI 需要设置 OPENAI_API_KEY）
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage


def build_model(provider: str, model_name: Optional[str] = None):
	provider = provider.lower()

	if provider == "openai":
		from langchain_openai import ChatOpenAI

		name = model_name or os.getenv("LLM_MODEL") or "gpt-4o-mini"
		return ChatOpenAI(model=name)

	if provider == "ollama":
		from langchain_ollama import ChatOllama

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
		default="What is the capital of France?",
		help="Prompt text",
	)
	args = parser.parse_args()

	model = build_model(args.provider, args.model)
	messages = [HumanMessage(args.prompt)]
	result = model.invoke(messages)
	content = getattr(result, "content", result)
	print(content)


if __name__ == "__main__":
	main()
