from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# You can check more available models in the OpenAI documentation.
model= ChatOpenAI(model_name="gpt-4", temperature= 0)       # the 'temperature' parameter controls how creative or focused the responses are. It ranges from 0 to 2.

result= model.invoke("What is the capital of India?")

print(result)
''' Now this time our 'result' will not be a plain text as we got in llm. Instead, it will be a ChatMessage object that contains additional metadata along with the text response.
Our text response can be accessed using 'result.content'. '''

print(result.content)