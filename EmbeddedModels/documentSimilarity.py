from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=300)

documents = [
    "The future of artificial intelligence and how AI could change the way humans work, learn, and communicate.",
    
    "Why some people can remember dreams vividly while others forget them within minutes of waking up.",
    
    "How social media algorithms influence what people believe, watch, and talk about without them realizing it.",
    
    "The possibility of humans living on Mars and the biggest technological and biological challenges we would face.",
    
    "Why time seems to pass faster as we get older and what psychology and neuroscience say about our perception of time."
]

query="I am depressed due to social media comparison"


doc_embeddings = embedding.embed_documents(documents) # o/p 5 vector with 300 dimentions

query_embedding = embedding.embed_query(query) # o/p 1 vector with 300 dimension 

similarity_scores = cosine_similarity([query_embedding],doc_embeddings)[0] # it gives similarity scores


# find the highest score index 
highest = similarity_scores[0]
index = 0

for i in range(len(similarity_scores)):
    if similarity_scores[i] > highest:
        highest = similarity_scores[i]
        index = i


print(query)
print(documents[index])

print("The above answer have a similarity score of ", highest)