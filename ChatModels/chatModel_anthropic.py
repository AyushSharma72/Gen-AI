from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()
model = ChatAnthropic(model="claude-fable-5")
result = model.invoke("are you better than chatgpt only answer in yes and no")

print(result.content)