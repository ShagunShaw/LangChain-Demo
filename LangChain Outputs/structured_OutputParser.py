from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task= "text-generation"
)

model = ChatHuggingFace(llm=llm)

schema= [
    ResponseSchema(name="fact 1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact 2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact 3", description="Fact 3 about the topic"),
]

parser= StructuredOutputParser.from_response_schemas(schema)

template= PromptTemplate(
    template= "Give me 3 interesting facts about {topic}. \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain= template | model | parser

result= chain.invoke({"topic": "space exploration"})

print(result)       # {'fact 1': '...', 'fact 2': '...', 'fact 3': '...'}
print(type(result)) 


# Now the drawback of this approach is that we cannot validate the fields in our output that are coming from the model, for that we can use PydanticOutputParser.