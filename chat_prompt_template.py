from langchain_ollama.chat_models import ChatOllama

from langchain.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Answer the following question: {question}")
])

llm = ChatOllama(model="llama3.2:latest")


chain = prompt | llm   # Chain

question='Who invented Agni (missile)'

response = chain.invoke({"question": question})

print(response.content)

