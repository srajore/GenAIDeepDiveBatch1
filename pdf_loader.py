from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('NOTES_Git.pdf')

documents = loader.load()

print(len(documents))

print(documents[0].metadata)

print(documents[0].page_content)