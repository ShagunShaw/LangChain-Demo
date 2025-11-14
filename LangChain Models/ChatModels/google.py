from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(model_name="gemini-1", temperature=1.6) 
result= model.invoke("What is the capital of India?")

print(result)       # Again the 'result' will be a ChatMessage object and not a plain text.
print(result.content)