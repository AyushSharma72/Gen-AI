from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

# chat template
chat_history = []
with open("chat_database.txt") as f:
     chat_history.extend(f.readline())

query = input("Enter your query") 

chat_template = ChatPromptTemplate(
    [
SystemMessage(content="you are the human being"),
MessagesPlaceholder(variable_name="chat_history"), # adding history here 
("human", "{query}")
    ]
)

result = chat_template.invoke({"chat_history":chat_history,"query":query})

print(result)


