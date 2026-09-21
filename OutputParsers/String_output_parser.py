from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",task ="text-generation")

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(template='Write a Note on {topic}',input_variables=['topic'])

template2 = PromptTemplate(template='generate 6 line summary of this \n {text}',input_variables=['text'])

parser = StrOutputParser()

chain  = template1 | model | parser | template2 | model | parser  # this is a chain and the output of one becomes the input of later

result = chain.invoke({'topic':'What skills are most important in the AI era which keeps the developer valuable'})

print(result)