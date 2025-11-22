# !pip install langchain langchain-core langchain-commmunity langchain_experimental pydantic duckduckgo-search

# DuckDuckGo Search (Built-in Tool)
from langchain_community.tools import DuckDuckGoSearchRun
search_tool= DuckDuckGoSearchRun()
result = search_tool.run("ipl news live")
print(result)



# Shell Tool ( Built-in Tool)
from langchain_community.tools import ShellTool
shell_tool = ShellTool()
output = shell_tool.run("whoami")       # can run any shell scripting command
print(output)


# There are many other built-in tools available in langchain_community.tools module. Check them out from the langhain_community.tools documentation.



# Custom Tool (using @tool decorator)
from langchain_core.tools import tool

# Step 1: Define a function
def multiply(a, b):
    """Multiply two numbers."""         # Not necessary but good practice to add docstring so that the LLM knows what this function does
    return a * b

# Step 2:  Add type hints
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Step 3: Add tool decorator
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Step 4: Use the tool
result = multiply.invoke({"a": 5, "b": 7})

print(result)  

print(multiply.name) 
print(multiply.description)  
print(multiply.args)  



# Custom Tool (using StructuredTool & Pydantic model)
"""
In this, we create a tool using well-defined Structured schema with Pydantic model, which helps in better validation and understanding of the tool's input parameters.
"""

fromlangchain.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")

def multiply_func(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

multiply_tool = StructuredTool.from_function(
    func= multiply_func,
    name= "multiply",
    description= "Multiply two numbers.",
    args_schema= MultiplyInput
)

result2= multiply_tool.invoke({"a": 6, "b": 8})

print(result2)

print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)



# Custom Tool (using BaseTool class)
"""
BaseTool is an abstract base class for all tools in LangChain. It defines the core structure and interface that all tools must follow.

All other tool types like @tool, StructuredTool, etc., are built on top of BaseTool.
"""
from langchain.tools import BaseTool
from typing import Type, Field

class MultiplyInputModel(BaseModel):
    a: int= Field(required=True, description="The first number to multiply")
    b: int= Field(required=True, description="The second number to multiply")


class MultiplyTool(BaseTool):
    name = "multiply"
    description = "Multiply two numbers."
    args_schema: Type[BaseModel] = MultiplyInputModel

    def _run(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b


multiply_tool2 = MultiplyTool()
result3 = multiply_tool2.invoke({"a": 9, "b": 4})

print(result3)

print(multiply_tool2.name)
print(multiply_tool2.description)