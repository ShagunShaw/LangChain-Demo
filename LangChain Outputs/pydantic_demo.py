# pip install pydantic

from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    firstName: str
    lastName: str = "Doe"       # We can also set default values like this
    email: EmailStr         # a built-in type for email validation
    age: Optional[int] = None       # Optional field. It is necessary to specify the default value in an Optional Field so that in case if it's value is not provided so it can use the default value
    cgpa: float= Field(gt=0.0, lt=10.0, default= 5, description="A decimal value represnenting the CGPA of student")   # Field with constraints: cgpa must be between 0.0 and 10.0


new_student = Student{          
    'firstName': "John",
    'email': "abc@example.com",
    'age': '32',        # Now pydantic is smart enough to convert this string to an integer, but if you pass  something non-convertible like "thirty-two", it will raise a validation error.
    'cgpa': 8.5
}

student1= Student(**new_student)

print(student1)     # This 'student1' is a pydantic object

student_dict= student1.dict()
print(student_dict)   # This 'student_dict' is now a python dictionary

student_json= student1.json()
print(student_json)   # This 'student_json' is a JSON string representation of the student