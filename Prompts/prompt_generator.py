from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["user_input", "tone"],
    template="""
Describe the following topic in no more than 6 lines.

Topic: "{user_input}"

Rules:
- Keep the explanation simple and easy to understand.
- Include at least one example.
- Keep the overall tone {tone}.
- Do not exceed 6 lines.
""",
validate_template=True #it checks if all the inputs are correctly given or not 
)


prompt_template.save('template.json') # save the template as json format