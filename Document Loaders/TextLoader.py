from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_prsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()  

model= ChatOpenAI(model_name= "gpt-3.5-turbo", temperature= 0)

prompt= PromptTemplate(
    template= "Write a summary for the following poem - \n {poem}",
    input_variables= ["poem"],
)

parser= StrOutputParser()

loader= TextLoader("example.txt", encoding= "utf-8")   # Ensure this file exists in your working directory

docs= loader.load()     # loads a file → it returns a list with only one Document object inside it, and the content of that object is the content of the file, and can be accessed via the 'docs[0].page_content'

print(type(docs))          # Will print: <class 'list'>
print(type(docs[0]))       # Will print: <class 'langchain.schema.document.Document'>


chain= prompt | model | parser
result= chain.invoke({'poem': docs[0].page_content})

print("Summary of the poem:")
print(result)