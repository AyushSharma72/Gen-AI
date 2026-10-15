from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

text = """Artificial Intelligence is transforming the way people work, learn, and communicate. Many companies use AI to automate repetitive tasks, analyze large amounts of data, and improve decision-making. Machine learning is a branch of artificial intelligence that allows computers to learn patterns from data without being explicitly programmed for every task.
Natural Language Processing, also known as NLP, helps computers understand and process human language. Applications of NLP include chatbots, language translation, sentiment analysis, and text summarization. Modern language models can generate human-like responses, answer questions, and assist developers with writing code.

Data plays an important role in building reliable AI systems. Before data can be used, it often needs to be collected, cleaned, and transformed. Data engineers build pipelines that move information from different sources into databases, data warehouses, and data lakes. They also ensure that the data is accurate, consistent, and available when needed.

Text splitting is an important step in many AI applications, especially retrieval-augmented generation systems. Large documents are divided into smaller chunks so that they can be processed efficiently. Each chunk can be converted into a numerical representation called an embedding. These embeddings help retrieve relevant information when a user asks a question.

Chunk size and chunk overlap affect the quality of text retrieval. Smaller chunks can provide more focused information, while larger chunks preserve more context. Overlap allows neighboring chunks to share some text, reducing the chance that important information is lost at chunk boundaries. The best configuration depends on the document type and the requirements of the application."""


file_path = Path(__file__).parent / "CrossCanals_Concurrency_Deep_Dive.pdf"
loader = PyPDFLoader(str(file_path)) # define loader 
doc = loader.load() # load the pdf in the doc variable

splitter  = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap = 0,  # keep some info similar between two chunks 
    separator=" "
)
# result = splitter.split_text(text)
result = splitter.split_documents(doc)
print(result[0].page_content) 


# output str 
# data = [
#     Document(
#         page_content="Artificial Intelligence is transforming the world.",
#         metadata={"source": "example.pdf", "page": 0}
#     ),
#     Document(
#         page_content="Machine learning helps computers learn from data.",
#         metadata={"source": "example.pdf", "page": 1}
#     )
# ]