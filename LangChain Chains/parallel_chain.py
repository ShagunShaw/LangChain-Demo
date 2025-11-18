from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain.schema.runnable import RunnableParallel      # For parallel chains

load_dotenv()

model1= ChatOpenAI()

model2= ChatAnthropic(model_name="claude-3-7-sonnet-20250219")

prompt1= PromptTemplate(
    template= "Generate short and simple notes from the following text.\n {text}",
    input_variables= ["text"]
)

prompt2= PromptTemplate(
    template= "Generate 5 short question answers from the following text.\n {notes}",
    input_variables= ["notes"]
)

prompt3= PromptTemplate(
    template= "Merge the provided notes and quiz into a simple document.\n notes-> {notes} and quiz-> {quiz}",
    input_variables= ["notes", "quiz"]
)

parser= StrOutputParser()


parallel_chain= RunnableParallel({          # Parallel chain creation
    'notesWalaChain': prompt1 | model1 | parser,
    'quizWalaChain': prompt2 | model2 | parser
})

merge_chain= prompt3 | model1 | parser      # Merging chain creation 

chain= parallel_chain | merge_chain   # Combining parallel and merging chains

my_text= '''
The Eiffel Tower is one of the most recognizable structures in the world, located in Paris, France. It was constructed as the entrance arch for the 1889 World's Fair, held to celebrate the 100th anniversary of the French Revolution. The tower was designed by the engineer Gustave Eiffel and stands at a height of 324 meters (1,063 feet). It was initially met with criticism from some of Paris's leading artists and intellectuals but has since become a global cultural icon of France and one of the most-visited paid monuments in the world. The Eiffel Tower is made of wrought iron and weighs approximately 10,100 tons. It has three levels for visitors, with restaurants on the first and second levels and an observation deck on the third level, offering panoramic views of Paris.
'''

result= chain.invoke({"text": my_text})

print(result)


chain.get_graph().print_ascii()  # To visualize the full chain structure in ASCII format