from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# chat_template = ChatPromptTemplate(
#     [SystemMessage(content="you are the expert of {domain}"),
#      HumanMessage(content="Explain about this {topic}")
#     ],validate_template=True)

chat_template = ChatPromptTemplate(
    [
        ("system", "You are the expert of {domain}"),
        ("human", "Explain about this {topic}")
    ],
   
)

domain_input = input("enter the domain ")
topic_input = input("enter the topic ")

prompt = chat_template.invoke({"domain":domain_input,"topic":topic_input})

print(prompt)

# chat prompt template is used to create chats templates when we are creating a chat bot and history is required to fit placeholder in multi turn messages 