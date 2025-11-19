from langchain.text_splitter import RecursiveCharacterTextSplitter

# This 'RecursiveCharacterTextSplitter' is better from 'CharacterTextSplitter' because it captivates the semantic meaning of the text while splitting it into smaller chunks.

# Now this 'RecursiveCharacterTextSplitter' does not split text based on chunks size only. It tries to split based on natural language structure like paragraphs, sentences, etc. 
# Firstly, it will try to split text by paras (\n\n), then by new lines (\n), then by periods (.), then by spaces, and finally by characters if necessary, maintaining the chunk size and overlap constraints.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0
)

text= '''Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. 
It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text. All the Lorem Ipsum generators on the Internet tend to repeat predefined chunks as necessary, making this the first true generator on the Internet. It uses a dictionary of over 200 Latin words, combined with a handful of model sentence structures, to generate Lorem Ipsum which looks reasonable. The generated Lorem Ipsum is therefore always free from repetition, injected humour, or non-characteristic words etc.
'''

chunks= splitter.split_text(text)

print(chunks)
print(len(chunks))