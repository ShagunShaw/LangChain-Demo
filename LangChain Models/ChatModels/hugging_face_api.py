# Here I am using Hugging Face Inference API to access a model, instead of downloading it locally.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()  

llm= HuggingFaceEndpoint(
    repo_id= "moonshotai/Kimi-K2-Thinking",
    task= "text-generation"
)

model= ChatHuggingFace(llm= llm)

result= model.invoke("What is the capital of India?")       # Again, this 'result' is a ChatResult object rather than a plain text.

print(result.content)