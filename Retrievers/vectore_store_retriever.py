from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

documents= [
    Document(page_content="This is a sample document about AI."),
    Document(page_content="This document discusses machine learning."),    
    Document(page_content="Natural language processing is a fascinating field."),
    Document(page_content="Deep learning techniques are widely used in AI."),
]

# Initialize the embedding model
embedding_model = OpenAIEmbeddings()

# Create a Chroma vector store from the documents
vector_store = Chroma.from_documents(documents, 
embedding=embedding_model,
collection_name="ai_documents")     # a new folder named 'ai_documents' will be created in the current directory


# Convert vectore store to a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})       # retrieve top 2 similar documents

query = "Tell me about AI and machine learning."
results= retriever.invoke(query)


for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content}")