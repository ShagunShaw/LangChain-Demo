import random

class NakliLLM:     # Creating a dummy LLM for now
    def __init__(self):
        print("LLM created")

    def predict(self, prompt):
        response_list= [        # dummy responses as we are not using any real model for now
            "Delhi is the capital of India.",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence."
        ]

        return {"response": random.choice(response_list)}


llm= NakliLLM()
print(llm.predict("What is the capital of India?"))



class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)


template= NakliPromptTemplate(
    template= "Write a {length} poem about {topic}",
    input_variables= ["length", "topic"]
)

print(template.format({"length": "short", "topic": "India"}))


# Now integrating both LLM and PromptTemplate (usng our defined classes)

prompt= template.format({"length": "short", "topic": "India"})

llm2= NakliLLM()
res= llm2.predict(prompt)

print(res)



class NakliLLMChain:
    def __init__(self, llm, prompt_template):
        self.llm = llm
        self.prompt_template = prompt_template

    def run(self, input_dict):
        prompt = self.prompt_template.format(input_dict)
        result= self.llm.predict(prompt)
        return result['response']

template2= NakliPromptTemplate(
    template= "Write a {length} poem about {topic}",
    input_variables= ["length", "topic"]
)

llm3= NakliLLM()

chain= NakliLLMChain(llm= llm3, prompt_template= template2)
output= chain.run({"length": "short", "topic": "India"})
print(output)