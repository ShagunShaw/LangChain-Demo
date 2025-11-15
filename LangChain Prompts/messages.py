from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI 

load_dotenv()

model = ChatGoogleGenerativeAI(model_name="gemini-1", temperature=1.6)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about LangChain."),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)