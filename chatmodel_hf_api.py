from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv(".env",override=True)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

try:
    result = model.invoke("What is the capital of India")
    print(result.content)
except Exception as e:
    print(f"Error: {e}")
    print("Make sure HUGGINGFACE_API_TOKEN is set in your .env file")