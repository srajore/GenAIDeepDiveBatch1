from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader('NOTES_Git.pdf')

documents = loader.load()


splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    separator=''
)

result = splitter.split_documents(documents)

print(result[0].page_content)