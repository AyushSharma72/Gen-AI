from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",task ="text-generation")

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()
template = PromptTemplate(template="give me the name age and city and movie of any movie character who struggles to get a job \n {format_instructions}",
    input_variables=[],
    partial_variables={'format_instructions':parser.get_format_instructions()})


prompt = template.format()

chain = template | model  | parser

result = chain.invoke({})

print(result)


# json output parser does not enforce a schema