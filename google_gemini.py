from langchain_google_genai import ChatGoogleGenerativeAI

#from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv


load_dotenv(override=True)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    #api_key=''  # Replace with your actual API key
)

response = llm.invoke("What is the capital of France?")

print(response.content)

