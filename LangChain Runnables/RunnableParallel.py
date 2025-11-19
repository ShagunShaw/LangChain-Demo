from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

prompt1= PromptTemplate(
    template= "generate a tweet about {topic}",
    input_variables= ["topic"]
)

prompt2= PromptTemplate(
    template= "generate a LinkedIn post about {topic}",
    input_variables= ["topic"]
)

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence([prompt1, model, parser]),
    'linkedin': RunnableSequence([prompt2, model, parser])
})

result = parallel_chain.invoke({"topic": "artificial intelligence"})        # Now this 'topic' input is shared across both chains

print(result)       # our 'result' will be a dictionary with keys 'tweet' and 'linkedin' containing respective outputs

print("\nTweet:\n", result['tweet'])
print("\nLinkedIn Post:\n", result['linkedin'])