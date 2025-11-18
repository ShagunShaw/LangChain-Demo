from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm= OpenAI(model= "gpt-3.5-turbo", temperature=0.7)

prompt= PromptTemplate(
    input_variables= ["topic"],
    template= "Select a catch blog title about {topic}."
)

topic= input("Enter a blog topic: ")

formatted_prompt= prompt.format(topic= topic)

blog_title= llm.predict(formatted_prompt)

print("Generated Blog Title:", blog_title)