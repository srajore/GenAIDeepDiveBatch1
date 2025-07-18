from langchain_ollama.chat_models import ChatOllama

llm = ChatOllama(model="llama3.2:latest")

response = llm.invoke("write a code to print hello in python")

print(response.content)
