from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional
from typing import Literal  # Literal is be used to restrict the output to certain values, it's like an 'enum' function that we use in other programming languages


# NOTE: TypedDict is only for structuring the output, it does not do any validation of the output. So, if the LLM does not follow the schema, TypedDict will not raise any errors. For example, if I have an age field in my schmea which is supposed to be a integer, but by mistake the LLM gives it as a String, so out TypeDict will not raise an error for that, it just defines the schema, not validates it. For validation, we can use Pydantic models.

load_dotenv()

model= ChatOpenAI()

# Generating our Output Schema
class Review(TypedDict):
    summary: str
    sentiment: Annotated[str, "The overall sentiment of the review, e.g., positive, negative, neutral"]         # Like this we can add descriptions to each field (if we want), so that our LLm models can better understand the output schema

structuredModel= model.with_structured_output(Review)

review= "The hardware is great, but the software feels bloated. There are too many pre-installed apps that I never use. Also, the UI looks outdated compared to other brands. Hoping for better software updates in the future."

result= structuredModel.invoke(review)

print(result)       # here no need to use 'result.content' as the output is already structured, so we can directly use 'result'

print(type(result))      # will print 'dict' showing that the output is structured as a dictionary

print(result['summary'])  
print(result['sentiment'])  




# Now let's work with a more complex reviews where we have to extract multiple things from the text

class DetailedReview(TypedDict):
    key_themes: Annotated[list[str], "A list of key themes discussed in the review"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg", "neutral"], "The overall sentiment of the review, e.g., positive, negative, neutral"]
    pros: Annotated[Optional[list[str]], "A list of pros mentioned in the review, if any"]
    cons: Annotated[Optional[list[str]], "A list of cons mentioned in the review, if any"]



review2= '''I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Shagun Shaw'''

structuredModel2= model.with_structured_output(DetailedReview)

result2= structuredModel2.invoke(review2)

print(result2)