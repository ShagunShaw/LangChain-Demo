from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat templeate
chat_template= ChatPromptTemplate([
    ('system', "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),  # placeholder for chat history (to be insserted before the the human message)
    ('human', "{query}")
])

# chat history (usually is loaded from database, but for now we will import it from chatHistory.txt file)

chatHistory= []
with open("chatHistory.txt") as f:
    chatHistory.extend(f.readlines())

# create prompt
final_prompt= chat_template.invoke({
    "chat_history": chatHistory,
    "query": "Where is my refund?"
})

# Thus we created our final prompt with chat history inserted in between system and human message. Now can use this prompt with any chat model (which you know how to do, so I am not repeating that here)