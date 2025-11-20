''' Wikipedia retrievers are tools in LangChain (and similar frameworks) that let you search and fetch content from Wikipedia as part of an LLM workflow

What they do

A Wikipedia retriever:

* Searches Wikipedia for a query.
* Fetches the most relevant pages.
* Returns cleaned text chunks so your LLM can use them for answering questions.
'''

# Run this code in COLAB instead (to save your memory)

# !pip install langchain chromadb faiss-cpu openai tiktoken langchain_openai langchain_community wikipedia 

from langchain_community.retrievers import WikipediaRetriever

retriever= WikipediaRetriever(
    top_k_results=2,    # number of relevant pages to fetch
    lang= "en"
)

query= "the geopolitical history of india and Pakistan from the perpespective of a chinese"

docs= retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"---------------Document {i+1}-----------------\n")
    print(doc.page_content)
    print("\n"+"-"*80+"\n")


'''
Q) how is wikipedia retriever different from a document loader?
A)  A Wikipedia retriever actively searches all Wikipedia pages on the internet for the most relevant pages based on your query and returns those relevant text chunks, while a document loader simply loads a specific, already-known document (PDF, TXT, website, etc.) without doing any search or relevance filtering.

In short: retriever = search + relevance, loader = load exactly what you point to.
'''