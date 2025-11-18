# RetrievalQAChain (in LangChain Runnables) is basically a ready-made pipeline that combines:

# A retriever → fetches relevant documents
# An LLM → answers a question using those documents
# A prompt template → formats everything correctly

# …and wraps all of this into one runnable object so you can call it with a single .invoke()


from langchain.document_loaders import TextLoader
from langchain.textsplitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

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

# Initialize the language model
llm = OpenAI(model= "gpt-3.5-turbo", temperature=0.7)

# Create the RetrievalQA chain (this is the key part)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ask a question (invoke the chain)
query= "What are the key takeaways from the document?"
response = qa_chain.run(query)

# Display the response
print("Response:", response)