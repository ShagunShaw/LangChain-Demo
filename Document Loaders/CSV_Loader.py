from langchain_community.document_loaders import CSVLoader

loader= CSVLoader(file_path='path/to/your/file.csv', encoding='utf-8')

documents = loader.load()

# Now if we had a very large CSV file, then we could have used 'documents = loader.lazy_load()' instead 

print(len(documents))  # Will print the number of documents loaded from the CSV file, i.e. if my file has 10 rows, it will print 10