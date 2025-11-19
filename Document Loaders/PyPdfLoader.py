# pip install pypdf ;  in your environment, before running this code

from langchain_community.document_loaders import PyPdfLoader

loader= PyPdfLoader("sample.pdf")   # Ensure you have a sample.pdf file in your directory

docs= loader.load()     # loads a PDF file → it returns a list of Document objects, each element in the list represents a page in the PDF. i.e if the PDF has 5 pages, the list will have 5 Document objects.

'''There are many more pdf loaders available in langchain, for example:
- UnstructuredPDFLoader
- PyMuPDFLoader
- PDFMinerLoader
- PDFPlumberLoader
- SlatePDFLoader, etc.
Each of these loaders has its own unique features and use-cases.
You can explore them in the official LangChain documentation.'''