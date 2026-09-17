from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32) 

result = embedding.embed_query("India is the capital of Delhi") # this converts the text into a 32 dimension vector 

print(str(result))

# A vector is a numerical representation of text in a multi-dimensional mathematical space, 
# where the position of the vector captures semantic information about the text.



# output 

# [-0.1563720703125, 0.2685546875, 0.0265655517578125
# , 0.256103515625, -0.146484375, 0.0753173828125, -0.027496337890625, 0.1163330078125, -0.09051513671875, -0.25048828125, -0.05517578125, -0.014495849609375, 
# -0.0787353515625, -0.2435302734375, -0.1861572265625, -0.194580078125, 0.06243896484375, 0.1444091796875, 0.136962890625, -0.1495361328125, 0.08978271484375,
#  -0.0220489501953125, 0.1075439453125, -0.0853271484375, -0.3828125, 0.53857421875, 0.113037109375, -0.08648681640625, -0.0254669189453125, 0.03472900390625,
#  0.173583984375, 0.019073486328125]