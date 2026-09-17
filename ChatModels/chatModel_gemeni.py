from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
result = model.invoke("All the government keep the people busy with sports and un usefull agends so that they never question the government so to rule the country")

print(result.content[0]["text"])