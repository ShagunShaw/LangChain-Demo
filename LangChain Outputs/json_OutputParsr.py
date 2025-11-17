from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain_core.output_parsers import JSONOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task= "text-generation"
)

model = ChatHuggingFace(llm=llm)

parser= JSONOutputParser()

template= PromptTemplate(
    template= "Give me the name, age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()} # Note since here we are specifying the format of the output, we need to pass it as a partial variable and not as an input variable
)

chain= template | model | parser

result= chain.invoke({})        # No input variables needed as we have none in the template, but we need to pass a dict then also, even if it is empty

print(result)       # {'name': 'John Doe', 'age': 30, 'city': 'New York'}
print(type(result)) # <class 'dict'> ; So we can access the fields directly like result['name'], result['age'], result['city']


# One drawback of this approach is that we cannot define our schema in this, like which format we want our json output to be in, or which fields we want. For that we can use StructuredOutputParser