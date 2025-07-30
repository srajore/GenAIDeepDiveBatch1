from langchain_openai import AzureOpenAI

from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv(override=True)

llm = AzureOpenAI(deployment_name="gpt-35-turbo-instruct", model_name="gpt-35-turbo-instruct")

prompt = ChatPromptTemplate.from_template("What is the future of AI in {foo}?")

chain = prompt | llm

response = chain.invoke({"foo": "India"})

print(response)