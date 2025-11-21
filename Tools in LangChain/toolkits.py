'''
A Toolkit is just a collection of related tools that serve a common purpose- packaged together for convenience and reusability.

For example, a "Data Analysis Toolkit" might include tools for data cleaning, visualization, and statistical analysis , so we bundle each of the tools of data analysis in a toolkit so that users can easily access all the tools they need for data analysis in one place.
'''

# Creating a custom toolkit
from langchain_core.tools import tool

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


class MathToolkit:
    def get_tools(self):            # The name of this method must be 'get_tools'
        return [add_numbers, multiply_numbers]

toolkit= MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
    print(f"Tool Name: {tool.name}, Description: {tool.description}")

# Using the toolkit
result_add = tools[0].invoke({"a": 3, "b": 5})
result_multiply = tools[1].invoke({"a": 4, "b": 6})

print(f"Addition Result: {result_add}")
print(f"Multiplication Result: {result_multiply}")