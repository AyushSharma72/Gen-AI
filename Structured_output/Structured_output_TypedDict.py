from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict
load_dotenv()


# from langchain_openai import ChatOpenAI 
# model = ChatOpenAI(model='gpt-5')

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",task ="text-generation")

model = ChatHuggingFace(llm=llm)

class Review(TypedDict): # this is the format in which we need the output from the llm
    summary:str
    sentiment:str


structured_output_model = model.with_structured_output(Review)

result = structured_output_model.invoke("I really like the mobile phone it runs very smoothly have high fps and never lags at all it is worth the money")


print(result)


# output by the hugging face deepseak model 
# {'summary': 'The mobile phone runs very smoothly with high FPS and never lags, making it worth the money.', 'sentiment': 'positive'}

# output by the open AI  gpt5 model 
# {'summary': 'Smooth, high-FPS performance with no lag; great value for money.', 'sentiment': 'positive'}