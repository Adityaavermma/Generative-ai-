from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
load_dotenv()
model=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')
prompt1= PromptTemplate(
    template="Generate a tweet about {text}",
    input_variables=["text"]
)
prompt2= PromptTemplate(
    template="Generate a linkdn post about {text}",
    input_variables=["text"]
)
parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})
result = parallel_chain.invoke({'text':'AI'})
print(result['tweet'])
print(result['linkedin'])
