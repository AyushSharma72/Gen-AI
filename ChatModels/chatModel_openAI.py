from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model='gpt-5')
# model = ChatOpenAI(model='gpt-5',max_completion_tokens=100,reasoning_effort="minimal")
# temperature is a paramneter which controls the creativity of the response of the model  temperature=0.2 its range lie between 0 to 2 

# max_completion_tokens is the number of words and characters we can specify the ai model to generate the answer within 


result = model.invoke("why human should live alone ?")


print(result.content)

# the result of this code is not simple text but it have a lot of data as output 