# this code works for python 3.12 and is thus not compatible with latest versions

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate       # Only for Dynamic Prompt Input

load_dotenv()


'''Static Prompt Input'''
user_input = input("Enter your prompt:")    

model = ChatGoogleGenerativeAI(model_name="gemini-1", temperature=1.6)

result= model.invoke(user_input)
print(result.content)      




'''Dynamic Prompt Input'''
templateText = "Please provide a brief summary of the following text: {text}. Make it concise and informative. And also provide {num} key takeaways in it."

textToSummarize = input("Enter the text to summarize:")
numKeyTakeaways = input("Enter number of key takeaways required:")

prompt = PromptTemplate(template=templateText, 
                        input_variables=["text", "num"],
                        validate_template=True)     # This will ensure that the template is valid and all input variables used in the template are provided in the 'input_variables' parameter.

# Format the prompt with actual values
formatted_prompt = prompt.invoke({
    "text": textToSummarize, 
    "num": numKeyTakeaways
})

model2 = ChatGoogleGenerativeAI(model_name="gemini-1", temperature=1.6)

result2= model2.invoke(formatted_prompt)
print(result2.content)

# Why is it suggested to use Dynamic Prompt instead of Static Prompt?
# Dynamic Prompts allow for greater flexibility and adaptability in generating responses. They enable users to customize the input based on specific needs or contexts (i.e. they just mention what they want, without having the need to hardcode the prompt every time as they used to do in case of static prompts), leading to more relevant and tailored outputs. This is particularly useful in scenarios where the input data may vary significantly, allowing for a more personalized interaction with the model.