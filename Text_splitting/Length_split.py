from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader(file_path=r"C:\Users\hp\OneDrive\Desktop\ML\gen ai\document loader\dl-curriculum.pdf")

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=' '

)
result=splitter.split_documents(loader.load())
print(result[0].metadata)