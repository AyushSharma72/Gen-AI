# mmr(Maximal Marginal Reteriver) is an information reterival algo designed to reduce the redundancy in the results while maintaining high relevance to the query 

# in regular similarity search you may get the documents that are 
# 1) All very similar 
# 2) repeating same info 
# 3) lacks diverse persepective 


# MMR 
# pick most relevant document first 
# then pcik the most relevent and least similar to the first one 
# and so on

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

load_dotenv()
docs = [
    Document(page_content="Python is a popular programming language used for AI and machine learning."),
    Document(page_content="JavaScript is widely used to build interactive websites and web applications."),
    Document(page_content="React is a JavaScript library used to build reusable user interface components."),
    Document(page_content="Node.js allows developers to run JavaScript on the server."),
    Document(page_content="Python supports data analysis, automation, and backend development."),
    Document(page_content="MongoDB is a NoSQL database that stores data in flexible JSON-like documents."),
    Document(page_content="Express.js is a Node.js framework used to build REST APIs."),
    Document(page_content="Machine learning models learn patterns from data to make predictions."),
    Document(page_content="HTML structures web pages, while CSS controls their appearance."),
    Document(page_content="Docker packages applications and their dependencies into containers.")
]

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./Reterivers/chroma_db",
    collection_name="fitness"
)



reteriver = vector_store.as_retriever(
    search_type="mmr",    # enables mmr search
    search_kwargs={ 
        "k":3,"lambda_mult":1  # k means top 3 documents and lambda unit means diversity balance
    }
)

# How lambda_mult works:

# Value	                    Behavior
# 1	                        Maximum relevance; no diversity preference (works just like similarity search)
# 0.5	                    Balance between relevance and diversity
# 0	                        Maximum diversity preference

query = "Which technologies are used for backend development?"

result = reteriver.invoke(query)

for i,doc in enumerate(result):
    print(f"\n --- Result {i+1} ---" )
    print(doc.page_content)