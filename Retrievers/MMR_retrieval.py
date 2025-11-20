''' MMR-> Maximal Marginal Relevance 

The main concept behind MMR is: How can we pick results that are both relevant to the query and are diverse from one another?
'''

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAIEmbeddings

docs= [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM applications."),
    Document(page_content="Chroma is used to store and retrieve document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps in retrieving diverse and relevant documents."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone and more."),
]

embedding_model = OpenAIEmbeddings()

vector_store = FAISS.from_documents(documents= docs, embedding= embedding_model)

# create a retriever with MMR
retriever = vector_store.as_retriever(search_type="mmr", 
search_kwargs={"k": 3, "lambda_mult": 0.5})  # k= top results, lambda_mult=1 (ranges between 0 and 1, higher value means more relevance, lower means more diversity)


query = "What is LangChain?"
results= retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content}")