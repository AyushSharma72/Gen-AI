from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

model = ChatOpenAI(model="gpt-5")

st.header("Research Tool")

user_input = st.text_input("Enter your query")
tone = st.selectbox("Select tone", ["positive", "negative","neutral"])

# prompt_template = PromptTemplate(
#     input_variables=["user_input", "tone"],
#     template="""
# Describe the following topic in no more than 6 lines.

# Topic: "{user_input}"

# Rules:
# - Keep the explanation simple and easy to understand.
# - Include at least one example.
# - Keep the overall tone {tone}.
# - Do not exceed 6 lines.
# """,
# validate_template=True #it checks if all the inputs are correctly given or not 
# )

prompt_template = load_prompt("./template.json")

if st.button("Summarize"):
    if user_input:

      #   prompt = prompt_template.invoke({ 
      #      "user_input":user_input,
      #      "tone":tone
      #  })

        # result = model.invoke(prompt)
        chain = prompt_template | model # create the template and call the model 
        # in single chain 
        result = chain.invoke({ 
            "user_input":user_input,
            "tone":tone
        })
        st.write("Calling model")
        st.write(result.content)
    else:
        st.warning("Please enter a query.")