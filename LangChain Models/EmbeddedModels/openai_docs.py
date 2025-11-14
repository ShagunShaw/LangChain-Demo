from langchain_openai import ChatOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = ChatOpenAIEmbeddings(model="text-embedding-3-large", dimension=32)

documents= [
    "Delhi is the capital of India.",
    "The capital of France is Paris.",
    "Kolkata is known as the city of joy."
]

vector= embedding.embed_documents(documents)
# Our vector will now be a list of 3 vectors, each of size 32.
# [ [0.12, -0.05, 0.33, ..., 0.07], [2.01, -1.05, 0.13, ..., -0.27], [1.12, 0.15, -0.23, ..., 0.47] ] (3 vectors of length 32)

print(str(vector))