from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
model=GoogleGenerativeAI(model='gemini-3.5-flash-lite')
prompt=PromptTemplate(
    template="write a summary for the following poem \n {text}",
    inputvariable=["text"]
)
parser=StrOutputParser()
loader = TextLoader(r"C:\Users\hp\OneDrive\Desktop\ML\gen ai\document loader\cricket.txt", encoding="utf8")
docs = loader.load()
print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)
chain= prompt | model | parser
result=chain.invoke({'text':docs[0].page_content})
print(result)
