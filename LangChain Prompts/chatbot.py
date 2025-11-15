# this code works for python 3.12 and is thus not compatible with latest versions

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage      # For maintaining chat history with roles

load_dotenv()

model = ChatGoogleGenerativeAI(model_name="gemini-1", temperature=1.6)

while True:
    user_input = input("Enter your prompt (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    result = model.invoke(user_input)
    print("AI:", result.content)


# Now while running the code, we will see a very crucial issue: After the first prompt, the model will not retain any memory of the previous interactions, so even though my next prompt is a continuation of the previous prompt, the model will not be able to understand that and will treat it as a completely new prompt. This is because we are not maintaining any chat history or context in this implementation. To overcome this, we need to implement a way to keep track of the conversation history and provide that context to the model with each new prompt.


chat_history = []

while True:
    user_input = input("Enter your prompt (or type 'exit' to quit): ")
    chat_history.append(user_input)
    if user_input.lower() == 'exit':
        break

    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("AI:", result.content)

print(chat_history)     # Now in this chat_history, we can see the entire converstaion, but we have no reference of who said what, which is a limitation as it may confuse our model in the long run. To overcome this, we will apply here some of the techniques as shown in messages.py file


chat_history2 = [
    SystemMessage(content="You are a helpful assistant")
]

while True:
    user_input = input("Enter your prompt (or type 'exit' to quit): ")
    chat_history2.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit':
        break

    result = model.invoke(chat_history2)
    chat_history2.append(AIMessage(content=result.content))
    print("AI:", result.content)

print(chat_history2)


# Now above all are examples of sending 'static' prompts to the model as we implicitly defined our system role as "assistant" and all our Human Messages are just user hardcoded inputs.  
# Now the demostration of 'dynamic' prompts in these cases is shown in chat_prompt_template.py file. So refer to it.