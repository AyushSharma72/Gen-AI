from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnableParallel

load_dotenv()
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",task ="text-generation")
model1 = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(template="generate short and simple note from the \n {text}",input_variables=["text"])
prompt2 = PromptTemplate(template="generate 5 questions and answers from the \n {text}",input_variables=["text"])
prompt3 = PromptTemplate(template="merge the {notes} and {quiz} into a single document",input_variables=["notes","quiz"])

parser = StrOutputParser()

paraller_chain = RunnableParallel({ # paraller chain
        'notes':prompt1 | model1 | parser, # will generate notes from the text 
        'quiz': prompt2 | model2 | parser  # will generate the quiz from the text
        })

merge_chain = prompt3 | model1 | parser

chain = paraller_chain|merge_chain

result = chain.invoke("""
Artificial intelligence is transforming the way people work, learn, and communicate across the world. Businesses use AI to automate repetitive tasks, analyze large amounts of data, and improve customer experiences. Students rely on AI-powered tools to understand complex concepts, practice coding, and generate ideas for projects. Healthcare professionals use AI to assist with medical imaging, predict diseases, and support faster decision-making. However, AI also brings challenges such as data privacy concerns, misinformation, and the need for responsible development. Companies must ensure that AI systems are transparent, fair, and secure. Individuals should learn how to use these tools effectively rather than depending on them blindly. As technology continues to evolve, skills like critical thinking, communication, and problem-solving become even more valuable. Developers are increasingly combining AI with web technologies, cloud computing, and automation platforms such as Docker, AWS, and workflow tools to build intelligent applications. The future of AI is likely to involve closer collaboration between humans and machines, where technology handles routine work while people focus on creativity, strategy, and innovation. Understanding AI today is becoming as important as learning to use computers was in previous decades, making continuous learning an essential habit for professionals and students alike.
""")

print(result)