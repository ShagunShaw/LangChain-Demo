# Now this semantic meaning based text spliiter is visually approaching concept-wise, but it is not perfectly structured yet and does not gives up to the mark result (at least for now), that is why it is kept under 'langchain_experimental' package. In future, many more works need to be done in it to increase the accuracy and once it is done and starts giving the result up the mark, then it's safe to use it in our codes too.

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter= SemanticChunker(
    embedding=OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

sample= """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

docs= text_splitter.create_documents([sample])

print(docs[0])
print(len(docs))