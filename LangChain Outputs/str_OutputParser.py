from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = OpenAI()

# Prompt 1
template1= PromptTemplate(
    template= "Write a detailed report on {topic}",
    input_variables= ["topic"]
)

# Prompt 2
template2= PromptTemplate(
    template= "Write a 5 line summary on the following text. \n {text}, in the tone: {tone}",
    input_variables= ["text", "tone"]
)

parser= StrOutputParser()       # Basically this Output Parser just returns the 'result.content' part of the LLM output, but here we are using Output Parser instead of directly accessing 'result.content', coz we want to demonstrate the use of Output Parsers in Langchain Chains (Pipelines).

chain = template1 | model | parser | (lambda text: {"text": text, "tone": "formal"}) | template2 | model | parser     # Creating a LangChain pipeline

result= chain.invoke({"topic": "Black Hole"})    

print(result)       # No need for 'result.content' here, just simply print the 'result'