from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documnets=[
    "delhi is the capital of india",
    "beging is the capital of china",
    "berlin is the capital of germany"
]

# result = embedding.embed_query("What is the capital of China?") # embeddings using the open source model 

result = embedding.embed_documents(documnets)  # embed documnets 

print(result)
print(len(result))
