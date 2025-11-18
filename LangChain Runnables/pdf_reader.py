from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

# Load the documents
loader= TextLoader("example.txt")       # Ensure 'example.txt' exists in the directory
documents = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert text into embeddings and store in FAISS vector store
vectorestore = FAISS.from_documents(docs, OpenAIEmbeddings())       # Vector DB stores embeddings for semantic search, plus original text so that when we try to retrieve those documents based on a query, you get the text, not the embeddings

# Create a retiever (fetches relevant documents based on a query)
retriever = vectorestore.as_retriever()

# Manually retrieves relevant documents for a sample query
query= "What are the key takeaways from the document?"
retrieved_docs = retriever.get_relevant_documents(query)

# Combine retrieved text into a single prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Initialize the language model
llm = OpenAI(model= "gpt-3.5-turbo", temperature=0.7)

# Manually pass retrieved text to the language model for generating a response
prompt= f"Based on the following information, answer the question: {query}\n\n{retrieved_text}"
response = llm.predict(prompt)

# Display the response
print("Response:", response)