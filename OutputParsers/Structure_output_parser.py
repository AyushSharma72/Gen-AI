from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",task ="text-generation")
model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(template='Write a Note on {topic}',input_variables=['topic'])

prompt1 = template1.invoke({'topic':'What skills are most important in the AI era which keeps the developer valuable'})

result1 = model.invoke(prompt1)

template2 = PromptTemplate(template='generate 6 line summary of this \n {text}',input_variables=['text'])

prompt2 = template2.invoke({'text':result1.content})

print("\n")

print("summary\n")

result2 = model.invoke(prompt2)

print(result2.content)