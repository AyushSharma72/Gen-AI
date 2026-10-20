# from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
load_dotenv()

docs = [
    Document(
        page_content="Inception is a science fiction thriller directed by Christopher Nolan, released in 2010. The story follows Dom Cobb, a skilled thief who steals secrets from people's dreams. He is given a mission to implant an idea into someone's mind. The movie explores dreams, reality, and the subconscious.",
        metadata={"title": "Inception", "genre": "Sci-Fi", "year": 2010}
    ),
    Document(
        page_content="The Dark Knight is a superhero crime thriller directed by Christopher Nolan, released in 2008. Batman faces the Joker, a criminal mastermind who creates chaos in Gotham City. The movie explores morality, justice, sacrifice, and the conflict between order and chaos.",
        metadata={"title": "The Dark Knight", "genre": "Action", "year": 2008}
    ),
    Document(
        page_content="Interstellar is a science fiction film directed by Christopher Nolan, released in 2014. A team of astronauts travels through a wormhole to find a habitable planet for humanity. The story explores space exploration, time dilation, love, and survival.",
        metadata={"title": "Interstellar", "genre": "Sci-Fi", "year": 2014}
    ),
    Document(
        page_content="3 Idiots is an Indian comedy-drama film directed by Rajkumar Hirani, released in 2009. The story follows three engineering students navigating academic pressure, friendship, and personal ambitions. The movie encourages curiosity, creativity, and learning beyond traditional education.",
        metadata={"title": "3 Idiots", "genre": "Comedy-Drama", "year": 2009}
    ),
    Document(
        page_content="The Shawshank Redemption is a drama film directed by Frank Darabont, released in 1994. It follows Andy Dufresne, a banker sentenced to life in prison, who develops a friendship with fellow inmate Red. The story explores hope, perseverance, friendship, and freedom.",
        metadata={"title": "The Shawshank Redemption", "genre": "Drama", "year": 1994}
    )
]


vector_store= Chroma(
    embedding_function=embedding_model,
   persist_directory="./vectorStore/chroma_db",
    collection_name="movies"
)

# vector_store.add_documents(docs) # generate the embeddigns and store them in the vectore store 

# data = vector_store.get()

# # print(data)


# {'ids': ['b47daf3d-e68a-4cfb-8894-33f648ac3e6f', '13e83e05-411b-4c96-8d5a-6dfc37e43350', 'e0f9dd55-04d4-459c-bdbb-8ca86b232d1d', '4ed6b890-4a8f-4695-992c-b29ec8f76dfd', '9c189279-39b6-494e-8f89-b684559f5520'], 'embeddings': None, 'documents': ["Inception is a science fiction thriller directed by Christopher Nolan, released in 2010. The story follows Dom Cobb, a skilled thief who steals secrets from people's dreams. He is given a mission to implant an idea into someone's mind. The movie explores dreams, reality, and the subconscious.", 'The Dark Knight is a superhero crime thriller directed by Christopher Nolan, released in 2008. Batman faces the Joker, a criminal mastermind who creates chaos in Gotham City. The movie explores morality, justice, sacrifice, and the conflict between order and chaos.', 'Interstellar is a science fiction film directed by Christopher Nolan, released in 2014. A team of astronauts travels througha wormhole to find a habitable planet for humanity. The story explores space exploration, time dilation, love, and survival.', '3 Idiots is an Indian comedy-drama film directed by Rajkumar Hirani, released in 2009. The story follows three engineering students navigating academic pressure, friendship, and personal ambitions. The movie encourages curiosity, creativity, and learning beyond traditional education.', 'The Shawshank Redemption is a drama film directed by Frank Darabont, released in 1994. It follows Andy Dufresne, a banker sentenced to life in prison, who develops a friendship with fellow inmate Red. The story explores hope, perseverance, friendship, and freedom.'], 'uris': None, 'included': ['metadatas', 'documents'], 'data': None, 'metadatas': [{'year': 2010, 'genre': 'Sci-Fi', 'title': 'Inception'}, {'title': 'The Dark Knight', 'genre': 'Action', 'year': 2008}, {'year': 2014, 'genre': 'Sci-Fi', 'title': 'Interstellar'}, {'genre': 'Comedy-Drama', 'year': 2009, 'title': '3 Idiots'}, {'genre': 'Drama', 'title': 'The Shawshank Redemption', 'year': 1994}]}



# search 

results = vector_store.similarity_search(
    query="which movie is about death",  # this will genrate an embedding and compare this with other genrated embeddings 
    k=1    ## default 4 
)

# print(results[0].metadata["title"])
print(results)

# o/p
# [Document(id='13e83e05-411b-4c96-8d5a-6dfc37e43350', metadata={'genre': 'Action', 'year': 2008, 'title': 'The Dark Knight'}, page_content='The Dark Knight is a superhero crime thriller directed by Christopher Nolan, released in 2008. Batman faces the Joker, a criminal mastermind who creates chaos in Gotham City. The movie explores morality, justice, sacrifice, and the conflict between order and chaos.')]


result_with_score = vector_store.similarity_search_with_score(  # this also sends the similarity score with the data we get above
    query="which movie is about heroism",  # this will genrate an embedding and compare this with other genrated embeddings 
    k=2   ## default 4 
)


print(result_with_score)

# o/p

# [(Document(id='9c189279-39b6-494e-8f89-b684559f5520', metadata={'year': 1994, 'genre': 'Drama', 'title': 'The Shawshank Redemption'}, page_content='The Shawshank Redemption is a drama film directed by Frank Darabont, released in 1994. It follows Andy Dufresne, a banker sentenced to life in prison, who developsa friendship with fellow inmate Red. The story explores hope, perseverance, friendship, and freedom.'), 1.2865755558013916), (Document(id='13e83e05-411b-4c96-8d5a-6dfc37e43350', metadata={'year': 2008, 'title': 'The Dark Knight', 'genre': 'Action'}, page_content='The Dark Knight is a superhero crime thriller directed by Christopher Nolan, released in 2008. Batman faces the Joker, a criminal mastermind who creates chaos in Gotham City. The movie explores morality,justice, sacrifice, and the conflict between order and chaos.'), 1.392290472984314)]
