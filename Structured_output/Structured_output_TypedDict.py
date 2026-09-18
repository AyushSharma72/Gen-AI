from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional
load_dotenv()


# from langchain_openai import ChatOpenAI 
# model = ChatOpenAI(model='gpt-5')

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",task ="text-generation") 

model = ChatHuggingFace(llm=llm)

class Review(TypedDict): # this is the format in which we need the output from the llm
    summary:Annotated[str,"A brief summary of the review"]
    sentiment:Annotated[str, "give the sentiment either positive, negative or netural"]# we can use Annotation to give clearity to the llm  
    key_features:Annotated[list[str],"Write down all the keys features discussed in the review"]
    cons:Annotated[Optional[list[str]],"Write down all the cons discussed in the review"]
    pros:Annotated[Optional[list[str]],"Write down all the pros discussed in the review"]
    name:Annotated[Optional[str], "Write the name of the reviewer"]

structured_output_model = model.with_structured_output(Review)

# result = structured_output_model.invoke("I really like the mobile phone it runs very smoothly have high fps and never lags at all it is worth the money")

# review 2 
# result = structured_output_model.invoke("the mobile is okay it sometimes give unnecessary isssues and become too hot when high cpu games are played but the overall phone is good according to the price range")


# review 3 for info extraction 
result = structured_output_model.invoke("I am Ayush Sharma and I've been using the OnePlus Nord 6, and for its price, it offers a really good overall experience. The phone feels smooth in day-to-day use, the display is vibrant with a high refresh rate, the performance is fast enough for multitasking, charging is impressively quick, and the cameras deliver decent results in good lighting. The biggest downside is that it can get noticeably hot during high-CPU gaming sessions and occasionally throws up minor random issues, although they aren't frequent enough to ruin the experience. Overall, if you're looking for a balanced mid-range phone with solid performance, a great display, and fast charging, the OnePlus Nord 6 is a worthwhile choice despite its heating and occasional software quirks")



print("=" * 50)
print("📱 OnePlus Nord 6 Review")
print("=" * 50)

print(f"\n📝 Summary:\n{result['summary']}")

print(f"\n✨ Key Features:\n{result['key_features']}")

print(f"\n😊 Sentiment:\n{result['sentiment']}")

print(f"\n✅ Pros:\n{result['pros']}")

print(f"\n❌ Cons:\n{result['cons']}")

print(f"\nReviewer Name\n{result['name']}") 

print("=" * 50)


# output by the hugging face deepseak model 
# {'summary': 'The mobile phone runs very smoothly with high FPS and never lags, making it worth the money.', 'sentiment': 'positive'}

# output by the open AI  gpt5 model 
# {'summary': 'Smooth, high-FPS performance with no lag; great value for money.', 'sentiment': 'positive'}

# output for other review 2 deepseek model
# {'summary': 'The mobile is okay overall. It sometimes gives unnecessary issues and becomes too hot when playing high CPU games, but the phone is good according to its price range.', 'sentiment': 'neutral'}


