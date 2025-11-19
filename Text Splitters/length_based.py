# Not widely used, as it sometimes produces non-intuitive results and fails to capture semantic meanings effectively.

from langchain.text_splitter import CharacterTextSplitter

splitter= CharacterTextSplitter(
    chunk_size=100,     # i.e. at every 100th character, make a split
    chunk_overlap=0,
    separator= ''
)

# For splitting a text
text= """In publishing and graphic design, Lorem ipsum is a placeholder text commonly used to demonstrate the visual form of a document or a typeface without relying on meaningful content. Lorem ipsum may be used as a placeholder before the final copy is available."""

result= splitter.split_text(text)   # 'result' will be a list of text chunks

print(f"Number of chunks: {len(result)}")
print(result)


# For splitting a document (like pdf)
from langchain_community.document_loaders import PyPDFLoader

loader2 = PyPDFLoader("sample.pdf")

doc = loader2.load()

result2= splitter.split_documents(doc)   # now here, doc is a list of documents (each page of the pdf is an individual document), so it will iterate through each document and split them based on the defined chunk size

print(f"Number of document chunks: {len(result2)}")
print(result2[0])
print(result2[0].page_content)  # to see the content of the first chunk

print(result2)