from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

pdf_directory = Path(__file__).parent

loader = DirectoryLoader(
    path=str(pdf_directory),
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs)) #834 number of pages in both pdfs 

print(docs[5].page_content) #5 number content