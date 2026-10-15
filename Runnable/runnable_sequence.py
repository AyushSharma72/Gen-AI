from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

llm = HuggingFaceEndpoint( repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",task="text-generation")
model = ChatHuggingFace(llm =llm)

# model = ChatOpenAI()
prompt = PromptTemplate(template="Tell me a unique joke on {topic} it must not be generic and famous but it must be good",input_variables=["topic"])

parser = StrOutputParser()

chain = RunnableSequence(prompt,model,parser) # this is a sequence of the steps that must run 
joke = chain.invoke({"topic":"javascript"})

print(joke)