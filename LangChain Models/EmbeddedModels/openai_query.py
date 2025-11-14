from langchain_openai import ChatOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = ChatOpenAIEmbeddings(model="text-embedding-3-large", dimension=32)
# Now in this, when we pass a sentence like "how are you", the entire sentence is embedded as one vector of size 32, not each word separately.
# "how are you" → [0.12, -0.05, 0.33, ..., 0.07] (length 32)

vector= embedding.embed_query("Delhi is the capital of India.")

print(str(vector))