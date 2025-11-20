'''
A Multi-Query Retriever is used to improve retrieval quality by generating multiple rewritten versions of the same user query, so the system can fetch more complete and diverse relevant documents. Instead of relying on one query (which might miss information), the LLM creates several semantic variations of the question, sends all of them to the retriever, and merges the results—reducing missed context and improving recall in a RAG pipeline.

Example usage:
If the query is "How can I stay healthy?";
The multi-query retriever might generate variations like "What are some tips for maintaining good health?", "How often should I exercise?" and "How do I improve my overall wellness?" The retriever then fetches documents for all these queries, ensuring a broader range of relevant information is retrieved.
'''

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_retrievers.multi_query import MultiQueryRetriever


all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Initialize the embedding model
embedding_model = OpenAIEmbeddings()

# Create a FAISS vector store from the documents
vector_store = FAISS.from_documents(documents= all_docs, embedding=embedding_model)

# Create the Multi-Query Retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    retriever= vector_store.as_retriever(search_kwargs={"k": 2})    # retrieve top 2 similar documents per query
)


query= "How to improve energy levels and maintain balance?"

# Retriever results
results = multi_query_retriever.invoke(query)


for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")