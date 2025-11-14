# Here I am downloading a hugging face model locally, instead of using the Hugging Face Inference API.
# In this code we will donwload the model to a local path and use it from there, instead of using the API.

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'F:\\Development Docs\\huggingface_cache'   # here we can set the path where we want to store the entire model folder

# DO NOT RUN THIS CODE IF YOU ARE USING CPU ONLY MACHINE, AS IT MAY REQUIRE A GPU TO RUN EFFICIENTLY, AND WILL ALSO CONSUME DISK SPACE TO STORE THE MODEL.
llm= HuggingFacePipeline.from_model_id(
    model_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task= "text-generation",
    pipeline_kwargs= {
        "max_new_tokens": 100,
        "temperature": 0.5
    }
)

model= ChatHuggingFace(llm= llm)

result= model.invoke("What is the capital of India?")
print(result.content)