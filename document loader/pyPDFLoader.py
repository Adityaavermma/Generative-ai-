from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader(r'C:\Users\hp\OneDrive\Desktop\ML\gen ai\document loader\dl-curriculum.pdf')
docs=loader.load()
print(docs)
print(docs[0].page_content)
print(docs[1].metadata)