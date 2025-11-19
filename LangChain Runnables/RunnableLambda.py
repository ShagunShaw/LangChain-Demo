# Runnable Lambda is used to convert any python function into a runnable object.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()

def word_count(text):
    return len(text.split())

prompt= PromptTemplate(
    template= "Write a joke about {topic}.",
    input_variables= ["topic"]
)

model= ChatOpenAI()

parser= StrOutputParser()

joke_gen_chain= RunnableSequence(prompt, model, parser)


# Now the puprose of this application is to get a joke from the llm and only count the number of words in the joke (and this time we dont want any joke's explanation)
parallel_chain= RunnableParallel({
'joke': RunnablePassthrough(),
'word_count': RunnableLambda(word_count)
})


fianl_chain= RunnableSequence(joke_gen_chain, parallel_chain)

result= fianl_chain.invoke({"topic": "chickens"})

print("Joke is: ", result['joke'])
print("Word count is: ", result['word_count'])