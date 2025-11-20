'''
Contextual Content Retriever is a specialized retriever that enhances the retrieval process by considering the context of the user's query. Instead of treating each query in isolation, it takes into account the surrounding context or previous interactions to fetch more relevant and coherent documents. This approach is particularly useful in conversational AI and complex information retrieval scenarios where understanding the context can significantly improve the quality of the results.

Exaample:

Suppose your document store contains info about different programming languages.

Conversation:
User: "I'm learning Python these days."
User (later): "What are its main features?"

A normal retriever hears only the new query “its main features?” → ambiguous → might return wrong docs.

A Contextual Content Retriever sees:

Previous message: “Python”

Current message: “its main features?”

So it retrieves documents about Python features, not Java or C++.
'''

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

docs= [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]


# Create a FAISS vector store from the documents
embedding_model = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embedding_model)

base_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Set up the LLM for compression
llm = ChatOpenAI(temperature=0)

compressor = LLMChainExtractor.from_llm(llm)

# Create the Contextual Content Retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever= base_retriever
)


# Example usage
query = "Tell me about photosynthesis."
results= compression_retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content}\n")


# NOTE: There are many more retriever options and configurations available in LangChain. So go and check them out in the documentation!