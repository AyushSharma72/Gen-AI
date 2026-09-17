from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",task ="text-generation")
# model = ChatHuggingFace(llm=llm)

llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K3",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="you are a maths expert"),
    HumanMessage(content="What would happen if maths was never invented")
]

result = model.invoke(messages)
print(result)
print("CONTENT:", repr(result.content))
messages.append(AIMessage(content=result.content))

print(messages)


# messsages are used to label the input of human and output of th ai so that it is easier fo the ai to distinguish the messages sent by human and the messages sent by the AI. 