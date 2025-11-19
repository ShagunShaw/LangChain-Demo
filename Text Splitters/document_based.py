from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text= """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")
"""

# NOTE: this time we are using Python language specific splitter, so instead of initialising our RecursiveCharacterTextSplitter by just 'RecursiveCharacterTextSplitter(...)', we are using 'RecursiveCharacterTextSplitter.from_language(...)' class method
splitter= RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

chunks= splitter.split_text(text)

print(chunks[0])
print(len(chunks))



# Let's now try this for a markdown text
text2= """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/student-tracker.git

"""

splitter2= RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=300,
    chunk_overlap=0
)

chunks2= splitter2.split_text(text2)

print(chunks2[0])
print(len(chunks2))