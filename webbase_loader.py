from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatOllama(model="llama3.2:latest")


prompt = PromptTemplate(
    template="Answer the following question \n{question} from the following text \n {text}",
    input_variables=["question", "text"]
)

parser = StrOutputParser()

url= "https://www.amazon.in/Daikin-Inverter-Copper-Filter-MTKM50U/dp/B09R4RYCJ4/ref=sr_1_3?_encoding=UTF8&content-id=amzn1.sym.58c90a12-100b-4a2f-8e15-7c06f1abe2be&dib=eyJ2IjoiMSJ9.LpujZ4uISPUK8sa_6yNGVY-3zoi-I7NYK-eHPsE7wGDDe5gR4wiXFNOAqexYtHRwhktNh6cntiQQEYFc77y5dBRisVemJuGFn8azR0KKEm5uYPGvPVbu2XG8IlAEucl1BoV8W4SygD5Lrn5Uvj70IqkbHDmy8R6mQ-GwxNL1GcebqYkXYRvlSLlRg1IzwgIiaeDhRKvEVxQKhGgkY51C5uWONkDYZkgqzsFaZsIrT6ijA0_hkEInzkw2-op2tZdY23McPNkJhC3lAa-xYn89N0LeUnqxOlYFLPv9miY4h-s.DSkCPeM7F7IH0afaGbhZQ8QFR0i75CDLGr_PIOWKqDM&dib_tag=se&pd_rd_r=6c8b2618-1739-4372-9414-58bc06002202&pd_rd_w=L4rjo&pd_rd_wg=R6Jie&qid=1753677002&refinements=p_85%3A10440599031&rps=1&s=kitchen&sr=1-3&th=1"

loader =WebBaseLoader(
    url,
    requests_kwargs={"headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"},
                     "verify": True
                     }
)

documents = loader.load()

chain = prompt | model | parser

response = chain.invoke({"question": "What is the price of the product?", "text": documents[0].page_content})

print(response)
