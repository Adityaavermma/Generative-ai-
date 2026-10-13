from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()
model=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')
prompt= PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"]
)
parser = StrOutputParser()
chain= prompt | model |parser
result=chain.invoke({'text':"Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection."})
print(result)
