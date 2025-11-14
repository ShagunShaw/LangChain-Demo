from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model= ChatAnthropic(model_name="claude-2", temperature=0) 
result= model.invoke("What is the capital of India?")

print(result)       # Again the 'result' will be a ChatMessage object and not a plain text.
print(result.content)