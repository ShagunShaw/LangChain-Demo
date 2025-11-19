import random
from abc import ABC, abstractmethod

class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass



class NakliLLM(Runnable):     # Creating a dummy LLM for now
    def __init__(self):
        print("LLM created")

    def invoke(self, prompt):
        response_list= [        # dummy responses as we are not using any real model for now
            "Delhi is the capital of India.",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence."
        ]

        return {"response": random.choice(response_list)}

    def predict(self, prompt):
        response_list= [        # dummy responses as we are not using any real model for now
            "Delhi is the capital of India.",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence."
        ]

        print("Warning: This method is going to be deprecated in future versions. Please use 'invoke' method instead.")

        return {"response": random.choice(response_list)}



class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        print("Warning: This method is going to be deprecated in future versions. Please use 'invoke' method instead.")

        return self.template.format(**input_dict)



class RunnableConnector(Runnable):
    def __init__(self, runnable_list):      # here we are using a runnable list coz we dont know how many runnables are going to be chained
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data



class NakliStrOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data['response']

# Use Case 1: Creating a simple chain
template= NakliPromptTemplate(
    template= "Write a {length} poem about {topic}",
    input_variables= ["length", "topic"]
)

llm= NakliLLM()

parser= NakliStrOutputParser()

chain = RunnableConnector([template, llm, parser])
output= chain.invoke({"length": "long", "topic": "India"})

print(output)


# Use Case 2: Combining two differet chains together
template1= NakliPromptTemplate(
    template= "Write a {joke} about {topic}",
    input_variables= ["joke", "topic"]
)

template2= NakliPromptTemplate(
    template= "Summarize the following joke: {text}",
    input_variables= ["text"]
)

llm2= NakliLLM()

parser2= NakliStrOutputParser()

chain1 = RunnableConnector([template1, llm2])

chain2 = RunnableConnector([template2, llm2, parser2])


combined_chain = RunnableConnector([chain1, chain2])
output2= combined_chain.invoke({"joke": "funny joke", "topic": "programming"})


print(output2)