from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import Field, BaseModel
from typing import Optional
load_dotenv()

# llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",task ="text-generation")

# model = ChatHuggingFace(llm=llm)
model = ChatOpenAI(model='gpt-5') 

class Review(BaseModel):
    summary:str=Field(description="A brief summary of the review")
    sentiment:str=Field(description="give the sentiment either positive, negative or netural")
    key_features:list[str]= Field(description="Write down all the keys features discussed in the review")
    cons:Optional[list[str]] =Field(default=None,description="Write down all the cons discussed in the review")
    pros:Optional[list[str]] = Field(default=None,description="Write down all the pros discussed in the review")
    name:Optional[str] = Field(default="Ayush",description="Write the name of the reviewer")

structured_output_model= model.with_structured_output(Review) # returns object


result = structured_output_model.invoke("I've been using the OnePlus Nord 6, and for its price, it offers a really good overall experience. The phone feels smooth in day-to-day use, the display is vibrant with a high refresh rate, the performance is fast enough for multitasking, charging is impressively quick, and the cameras deliver decent results in good lighting. The biggest downside is that it can get noticeably hot during high-CPU gaming sessions and occasionally throws up minor random issues, although they aren't frequent enough to ruin the experience. Overall, if you're looking for a balanced mid-range phone with solid performance, a great display, and fast charging, the OnePlus Nord 6 is a worthwhile choice despite its heating and occasional software quirks ")

print("=" * 50)
print("OnePlus Nord 6 Review")
print("=" * 50)

print(f"\nReviewer:\n{result.name}")

print(f"\nSummary:\n{result.summary}")

print(f"\nKey Features:\n{', '.join(result.key_features)}")

print(f"\nSentiment:\n{result.sentiment}")

print("\nPros:")
for pro in result.pros:
    print(f"• {pro}")

print("\nCons:")
for con in result.cons:
    print(f"• {con}")

print("=" * 50)


# output 
# ==================================================
# OnePlus Nord 6 Review
# ==================================================

# Reviewer:
# None

# Summary:
# Strong value mid-ranger with smooth daily performance, a vibrant high-refresh display, fast charging, and decent daylight cameras; main drawbacks are heating during heavy gaming and occasional minor software quirks.

# Key Features:
# Good price-to-performance value, Smooth day-to-day experience, Vibrant display with high refresh rate, Performance adequate for multitasking, Very fast charging, Decentcamera quality in good lighting, Heats up during high-CPU gaming, Occasional minor software issues/quirks

# Sentiment:
# positive

# Pros:
# • Smooth daily performance
# • Vibrant high-refresh display
# • Fast charging
# • Good multitasking performance
# • Decent cameras in good lighting
# • Balanced mid-range experience and value

# Cons:
# • Gets noticeably hot during high-CPU gaming
# • Occasional minor random/software issues
# ==================================================