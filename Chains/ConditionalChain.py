from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=llm)


# ---------------- Sentiment Schema ----------------

class ReviewSentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment of the review."
    )


sentiment_parser = PydanticOutputParser(pydantic_object=ReviewSentiment)


# ---------------- Classification Chain ----------------

sentiment_classification_prompt = PromptTemplate(
    template="""
Based on the review below, classify the sentiment as only one of these:
positive, negative, or neutral.

Review: {Review}

{format_instructions}
""",
    input_variables=["Review"],
    partial_variables={
        "format_instructions": sentiment_parser.get_format_instructions()
    }
)

sentiment_classifier_chain = (
    sentiment_classification_prompt
    | chat_model
    | sentiment_parser
)


# ---------------- Response Prompts ----------------

positive_response_prompt = PromptTemplate(
    template="Write an appropriate response to this positive feedback.\n\n{Review}",
    input_variables=["Review"]
)

negative_response_prompt = PromptTemplate(
    template="Write an appropriate response to this negative feedback.\n\n{Review}",
    input_variables=["Review"]
)

neutral_response_prompt = PromptTemplate(
    template="Write an appropriate response to this neutral feedback.\n\n{Review}",
    input_variables=["Review"]
)

# ---------------- Response Chains ----------------

positive_response_chain = positive_response_prompt | chat_model
negative_response_chain = negative_response_prompt | chat_model
neutral_response_chain = neutral_response_prompt | chat_model

review = "What a classic phone it is. I really like it. Thanks for the service OnePlus"

sentiment = sentiment_classifier_chain.invoke({
    "Review": review
}).sentiment

if sentiment == "positive":
    result = positive_response_chain.invoke({"Review": review})
elif sentiment == "negative":
    result = negative_response_chain.invoke({"Review": review})
else:
    result = neutral_response_chain.invoke({"Review": review})

print(result.content)





# # ---------------- Branch ----------------

# feedback_response_branch = RunnableBranch(
#     (lambda x: x["sentiment"] == "positive", positive_response_chain),
#     (lambda x: x["sentiment"] == "negative", negative_response_chain),
#     (lambda x: x["sentiment"] == "neutral", neutral_response_chain),
#     RunnableLambda(lambda _: "Could not determine the sentiment.")
# )


# # ---------------- Final Chain ----------------

# prepare_input = RunnableLambda(
#     lambda review: {
#         "Review": review,
#         "sentiment": sentiment_classifier_chain.invoke(
#             {"Review": review}
#         ).sentiment
#     }
# )

# final_chain = prepare_input | feedback_response_branch


# # ---------------- Run ----------------

# result = final_chain.invoke(
#     "What a classic phone it is. I really like it. Thanks for the service OnePlus 😘"
# )

# print(result)