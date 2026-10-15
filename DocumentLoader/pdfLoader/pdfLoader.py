# # it gives page_content and meta_data for every page it scans in the pdf 
# from pathlib import Path
# from langchain_docling import DoclingLoader
# file_path = Path(__file__).parent / "ayush_sharma_profile.pdf"

# loader = DoclingLoader(str(file_path))
# docs = loader.load()
# # print(docs)
# print(docs[1].page_content)
# # print(docs[0].metadata)



from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
file_path = Path(__file__).parent / "ayush_sharma_profile.pdf"
loader = PyMuPDFLoader(str(file_path))

docs = loader.load()

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)