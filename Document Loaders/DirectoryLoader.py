# A directory loader helps to load multiple files (txt, pdf, etc.) at a time from a specified directory.

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader= DirectoryLoader(
    path= './',  # path to the directory containing files,
    glob= '*.pdf' ,         # load all files that matches the given pattern 
    loader= PyPDFLoader  # loader class to use for loading the files. If it's a text file, use TextLoader, if it's a csv file, use CSVLoader, and more like this.
)

docs= loader.load()

print(len(docs))  
# Understand what it will print here. Suppose I have 3 pdf files in the specified directory; pdf 1 has 10 pages, pdf 2 has 5 pages, and pdf 3 has 7 pages. Now it will print 22 because each page of each pdf is considered as a separate document.



'''Problem with load() and solution with lazy_load(): 
In LangChain, the main issue with load() was that it loads all documents and all pages at once, which is fine for a single small PDF/text, but becomes very slow and memory-heavy when you have many PDFs with many pages. Because everything is processed upfront, you must wait a long time before you can start working, and large document sets can even crash due to high RAM usage. To fix this, LangChain introduced lazy_load(), which loads documents one at a time only when needed, making it much faster and more efficient for large or multiple PDFs.

So in such a case, when you have many files or large files to be loaded, use lazy_load() instead of load() like this:  docs= loader.lazy_load()
'''