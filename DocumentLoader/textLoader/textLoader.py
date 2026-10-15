from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
file_path = Path(__file__).parent / "text.txt"

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="zai-org/GLM-5.3",task ="text-generation")

model = ChatHuggingFace(llm=llm)

loader = TextLoader(str(file_path), encoding="utf-8")

prompt = PromptTemplate(template="What Ayush values in his life, aslo predict his future based on this data \n {data}",input_variables=["data"])

parser = StrOutputParser()


docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({"data":docs[0].page_content})


print(result)
# print(docs) # it have page content and meta data 


