from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",task ="text-generation")
model = ChatHuggingFace(llm=llm)

chat_history=[
    SystemMessage(content="you are a maths expert"),
 ] # for storing the memory 

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input)) # add current conversation in chat history 
    if user_input=='exit':
        break
    result = model.invoke(chat_history) 
    chat_history.append(AIMessage(content=result.content)) # save the ai response in chat history 
    print("AI: " ,result.content)


print(chat_history)