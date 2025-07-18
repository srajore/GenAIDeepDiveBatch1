from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

llm = ChatOpenAI()

response = llm.invoke("write a code to print hello in python")

print(response.content)