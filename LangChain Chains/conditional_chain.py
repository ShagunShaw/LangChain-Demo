from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

from langchain.schema.runnable import RunnableBranch     # For conditional chains
from langchain.schema.runnable import RunnableLambda     # For 

load_dotenv()

model1= ChatOpenAI()

parser= StrOutputParser()

class FeedBack(BaseModel):
    sentiment: Literal["positive", "negative"]= Field(description="The sentiment of    the feedback")

parser2= PydanticOutputParser(pydantic_object= FeedBack)

prompt1= PromptTemplate(
    template= "Classify the sentiment of the following feedback into positive or negative:\n{feedback}\n{format_instructions}",
    input_variables= ["feedback"],
    partial_variables= {"format_instructions": parser2.get_format_instructions()}
)

classifier_chain= prompt1 | model1 | parser2

prompt2= PromptTemplate(
    template= "Write an appropriate response to this positive feedback: \n {feedback}",
    input_variables= ["feedback"]
)

prompt3= PromptTemplate(
    template= "Write an appropriate response to this negative feedback: \n {feedback}",
    input_variables= ["feedback"]
)

conditional_chain1= prompt2 | model1 | parser
conditional_chain2= prompt3 | model1 | parser

# NOTE: classifier_chain outputs a FeedBack object, BUT after being passed to the next chain, the Branch receives a dictionary, not a Pydantic object. LangChain converts outputs internally to dict-like structures. So instead of accessing result.sentiment (which we use to access Pydantic Object values) in the next stage in the chain, we will use result['sentiment'] (the way we access dictionary values) in the lambda functions.

branch_chain= RunnableBranch(
    (lambda x: x['sentiment'].lower() == "positive", conditional_chain1),
    (lambda x: x['sentiment'].lower() == "negative", conditional_chain2),
    RunnableLambda(lambda x: "No valid sentiment found.")       # Default case, since this is not a part of any condition and is a sort of exception handling i.e when no valid condition is met, then it will return this default case.
)

final_chain= classifier_chain | branch_chain

result= final_chain.invoke({"feedback": "The product was great and I loved it!"})
print(result)

final_chain.get_graph().print_ascii()  # To visualize the full chain structure in ASCII format