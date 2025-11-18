# Here we replace the step-by-step LLM code of 'simple_llm_app.py' with an automated, reusable LLMChain pipeline

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm= OpenAI(temperature=0, model_name="gpt-3.5-turbo")

prompt= PromptTemplate(
    input_variables= ["topic"],
    template= "Select a catch blog title about {topic}."
)

chain= LLMChain(llm= llm, prompt= prompt)

topic= input("Enter a blog topic: ")
output= chain.run(topic)

print("Generated Blog Title:", output)