# In this code, we will use 'Chroma' as our vector store.
# Why Chroma? : Coz it is a lightweight, open-source vector database that is specially designed for local development and small to medium-scale production needs.

# Run this in cmd first:
# pip install langchain chromadb openai tiktoken pypdf langchain_openai langchain_community


from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Create some sample documents
doc1= Document(
    page_content="This is the content of IPL team one: Mumbai Indians.", 
    metadata={"source": "doc1.txt", "team": "Mumbai Indians"}
)

doc2= Document(
    page_content="This is the content of IPL team two: Chennai Super Kings.",
    metadata={"source": "doc2.txt", "team": "Chennai Super Kings"}
)

doc3= Document(
    page_content="This is the content of IPL team three: Royal Challengers Bangalore.",
    metadata={"source": "doc3.txt", "team": "Royal Challengers Bangalore"}
)

doc4= Document(
    page_content="This is the content of IPL team four: Kolkata Knight Riders.",
    metadata={"source": "doc4.txt", "team": "Kolkata Knight Riders"}
)

doc5= Document(
    page_content="This is the content of IPL team five: Delhi Capitals.",
    metadata={"source": "doc5.txt", "team": "Delhi Capitals"}
)

docs= [doc1, doc2, doc3, doc4, doc5]

vector_store= Chroma(
    embedding_function= OpenAIEmbeddings(),
    collection_name= "sample",  # table name
    persist_directory= "my_chroma_db"  # database name
)
# Now this database 'my_chroma_db' folder will be created in our current working directory.


# Add documents to the vector store
vector_store.add_documents(documents=docs, ids=["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt", "doc5.txt"])


# view documents
retrieved_docs = vector_store.get(include=["embeddings", "documents", 'metadatas'])

for doc in retrieved_docs:
    print(doc['documents'], doc['metadatas'])
    print(doc['embeddings'][:10])  # print first 10 values of the embedding vector
    print("---------------------------------")


# search similar documents
query = "Tell me about the IPL team from Mumbai."
similar_docs = vector_store.similarity_search(query, k=2)       # k= number of similar documents to retrieve
for doc in similar_docs:
    print(doc.page_content, doc.metadata)
    print("---------------------------------")


# search similar documents along with their similarity scores
similar_docs_with_scores = vector_store.similarity_search_with_score(query, k=2)
for doc, score in similar_docs_with_scores:
    print(doc.page_content, doc.metadata, "Score:", score)
    print("---------------------------------")


# metadata filtering during search
filter_criteria = {"team": "Chennai Super Kings"}
filtered_docs = vector_store.similarity_search_with_score(query= "", filter=filter_criteria)
for doc, score in filtered_docs:
    print(doc.page_content, doc.metadata, "Score:", score)
    print("---------------------------------")


# IMP: Update documents in the vector store
# Let's say we want to update the content of the document related to "Delhi Capitals".
updated_doc = Document(
    page_content="This is the UPDATED content of IPL team five: Delhi Capitals - The Rising Stars.",
    metadata={"source": "doc5.txt", "team": "Delhi Capitals"}
)

vector_store.delete(ids=["doc5.txt"])
vector_store.add_documents([updated_doc], ids=["doc5.txt"])


# IMP: Delete documents from the vector store
vector_store.delete(ids= ['doc3.txt', 'doc4.txt'])