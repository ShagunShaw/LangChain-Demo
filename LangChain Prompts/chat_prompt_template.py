# This code works for python 3.12 and is thus not compatible with latest versions

langchain_core.prompts import ChatPromptTemplate

chat_template= ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert."),
    ('human', "Explain the concept of {topic} in simple terms.")
])

# Now we can format the prompt with actual values
prompt= chat_template.invoke({
    "domain": "deep learning",
    "topic": "neural networks"
})

model = ChatGoogleGenerativeAI(model_name="gemini-1", temperature=1.6)
result= model.invoke(prompt)
print(result.content)


# Now there is one more concept of message placeholders. A message placeholder is a way to define a part of the message that can be filled in later.
# A code of message placeholder is given in message_placeholder.py file