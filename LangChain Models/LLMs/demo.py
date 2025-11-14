# Here we are using OPEN_AI

from langchain_openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()           

# Here, we don’t need to explicitly pass the API key in our code. As long as load_dotenv() has run, the 'OPENAI_API_KEY' is picked up automatically.
llm = OpenAI(model="gpt-3.5-turbo-instruct")  
response = llm("What is the capital of India?")     # Generate a response for the given prompt

print(response) 
'''Our result will be: 
The capital of India is New Delhi.'''