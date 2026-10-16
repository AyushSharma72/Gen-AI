from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text = """
Python is a programming language. It supports object-oriented programming.
Classes and objects are important concepts in Python.

PySpark is used to process large datasets across multiple computers.
It is widely used in data engineering and big data processing.

SQL is used to query relational databases.
It supports operations such as filtering, grouping, and joining data.
"""

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_amount=0.5,
    breakpoint_threshold_type="standard_deviation"
)

chunks = text_splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:\n{chunk}\n")
