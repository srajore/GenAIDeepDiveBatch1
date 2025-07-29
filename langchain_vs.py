from langchain.schema import Document
#from langchain.vectorstores import Chroma
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_core.prompts import PromptTemplate

from langchain.chains import RetrievalQA

document1= Document(
    page_content='Virat kohli is one of the most successful and consistant batsman in the world. He has scored more than 70 international centuries and is known for his aggressive batting style.',
    metadata={'team':'Royal Challengers Bangalore'}
)

document2 = Document(
    page_content='Rohit Sharma is a prolific run-scorer and one of India’s finest opening batsmen. He holds the record for the highest individual score in an ODI match with 264 runs and is known for his elegant batting style.',
    metadata={'team': 'Mumbai Indians'}
)

document3 = Document(
    page_content='Suryakumar Yadav is a dynamic middle-order batsman known for his explosive 360-degree batting style. He has been a key player for Mumbai Indians, consistently delivering match-winning performances.',
    metadata={'team': 'Mumbai Indians'}
)

document4 = Document(
    page_content='Jasprit Bumrah is one of the world’s premier fast bowlers, known for his unique bowling action and deadly yorkers. He has been a key figure in Mumbai Indians’ bowling attack in the IPL.',
    metadata={'team': 'Mumbai Indians'}
)

document5 = Document(
    page_content='Yuzvendra Chahal is a world class bowlers crafty leg-spinner known for his wicket-taking ability in the middle overs. He has been a standout performer for Rajasthan Royals, often turning matches with his variations.',
    metadata={'team': 'Rajasthan Royals'}
)


documents = [document1, document2, document3, document4, document5]

vector_store = Chroma(
    embedding_function=OllamaEmbeddings(model="granite-embedding:latest"),
    persist_directory='my_chroma_db',
    collection_name='sample_collection'
)

vector_store.add_documents(documents)

#print(vector_store.get(include=['embeddings','documents','metadatas']))

#llm = ChatOllama(model='llama3.2:latest')
llm = ChatOpenAI(model='gpt-4o-mini', max_tokens=500)
prompt_template = """
   
    You are an expert in cricket statistics and player analysis. Given the following documents about cricket players, answer the question based on the information provided.
    Context:{context}

    question: {question}    
    Answer:

    """

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

#query = "Who among these are the bowlers?"

query ="who all are bowlers?"

result = qa_chain.invoke({"query": query})

print("Answer:", result['result'])

print("Source Documents:",result['source_documents'])   