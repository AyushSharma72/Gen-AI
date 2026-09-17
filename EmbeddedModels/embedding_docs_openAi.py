from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
documnets=[
    "delhi is the capital of india",
    "beging is the capital of china",
    "berlin is the capital of germany"
]
embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=2) 

result = embedding.embed_documents(documnets)  # generate a vector of this doc 

print(str(result))

# output 
# [[-0.60107421875, 0.7998046875], [0.375, 0.927734375], [0.62158203125, 0.783203125]]