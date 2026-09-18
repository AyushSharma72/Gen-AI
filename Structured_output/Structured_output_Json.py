from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",task ="text-generation")

model = ChatHuggingFace(llm=llm)
# model = ChatOpenAI(model='gpt-5')

json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum":["positive","negative","neutral"],
      "description": "give the sentiment either positive, negative or netural"
    },
    "key_features": {
      "type": "array",
      "description": "Write down all the keys features discussed in the review",
      "items": {
        "type": "string"
      }
    },
  "pros": {
  "type": "array",
  "items": { "type": "string" },
  "description": "Write each pro as a separate list item."
},
"cons": {
  "type": "array",
  "items": { "type": "string" },
  "description": "Write each con as a separate list item."
},
    "name": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
     "default": None,
      "description": "Write the name of the reviewer"
    }
  },
  "required": [
    "summary",
    "sentiment",
    "key_features"
  ]
}

structured_output_model= model.with_structured_output(json_schema) # returns dictionary 


result = structured_output_model.invoke("I've been using the OnePlus Nord 6, and for its price, it offers a really good overall experience. The phone feels smooth in day-to-day use, the display is vibrant with a high refresh rate, the performance is fast enough for multitasking, charging is impressively quick, and the cameras deliver decent results in good lighting. The biggest downside is that it can get noticeably hot during high-CPU gaming sessions and occasionally throws up minor random issues, although they aren't frequent enough to ruin the experience. Overall, if you're looking for a balanced mid-range phone with solid performance, a great display, and fast charging, the OnePlus Nord 6 is a worthwhile choice despite its heating and occasional software quirks")


print(result)