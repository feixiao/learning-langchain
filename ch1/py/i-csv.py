from langchain_core.output_parsers import CommaSeparatedListOutputParser


# You can also use an LLM or chat model to produce output in other formats, such as CSV or XML. 
parser = CommaSeparatedListOutputParser()

response = parser.invoke("apple, banana, cherry")
print(response)
