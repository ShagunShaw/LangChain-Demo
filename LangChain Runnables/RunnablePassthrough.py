# In Runnable Passthrough, whatever input is given to our llm is returned as output without any modifications.

'''Now it's not that ki Runnable Passthrough is useless. In fact, it can be quite handy in certain scenarios. Lets's design one such case here on out "jokes" scenario. Earlier, we got the joke from llm and we just passed it to the next llm for the joke explanation. But what if we want to see the joke itself along with the explanation? In such cases, we can use Runnable Passthrough to simply pass the joke as it is to the next llm for explanation.'''


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

prompt1= PromptTemplate(
    template= "Write a joke about {topic}.",
    input_variables= ["topic"]
)

prompt2= PromptTemplate(
    template= "Explain the following joke: {joke}",
    input_variables= ["joke"]
)

model= ChatOpenAI()

parser= StrOutputParser()

joke_generator_chain= RunnableSequence(prompt1, model, parser)

parallel_chain= RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

full_chain= RunnableSequence(joke_generator_chain, parallel_chain)

result= full_chain.invoke({'topic': 'computers'})

print("Joke:", result['joke'])
print("Explanation:", result['explanation'])