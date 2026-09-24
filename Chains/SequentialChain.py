from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",task ="text-generation")
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(template="Generate the detailed sumary on {topic}",input_variables=["topic"])
prompt2 = PromptTemplate(template="Generate a 5 line summart on this text {'text'}",input_variables=["text"])

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser 

result = chain.invoke({'topic':"Just Like AI what are the other disrputions due to which people lost jobs"})

print(result)