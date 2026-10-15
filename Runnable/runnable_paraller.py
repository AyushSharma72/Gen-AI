# from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

llm = HuggingFaceEndpoint( repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",task="text-generation")
model = ChatHuggingFace(llm =llm)

# model = ChatOpenAI()
linkedinPrompt = PromptTemplate(template="Generate the caption for Linkedin post on the {linkedinTopic} ",input_variables=["linkedinTopic"])
twitterPrompt = PromptTemplate(template="Generate the caption for twitter post on the {twitterTopic} ",input_variables=["twitterTopic"])


parser = StrOutputParser()

chain = RunnableParallel( # execute two chains in parallel
    {
    "linkedinCaption":linkedinPrompt | model | parser,
    "twitterCaption":twitterPrompt | model | parser
    }
)  

result = chain.invoke({"linkedinTopic":"javascript descreasing popularity","twitterTopic":"java descreasing popularity"})

print(result['linkedinCaption'])
print(result['twitterCaption'])

# output
# {'linkedinCaption': '"Is JavaScript losing its edge? Explore the latest trends and insights on the decreasing popularity of JavaScript in the developer community. #javascript #programming #techindustry"', 'twitterCaption': 'Is Java losing its appeal? 📉 Despite being a widely-used language, recent data shows a decline in its popularity. What do you think could be the reason behind this trend? #Java #programming #techtrends'}