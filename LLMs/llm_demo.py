from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-5')

result = llm.invoke("which is the best Ai model")


print(result)