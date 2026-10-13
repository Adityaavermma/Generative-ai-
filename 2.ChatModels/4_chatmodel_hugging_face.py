from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-1.5B-Instruct",
)
model=ChatHuggingFace(llm=llm)
result=model.invoke("what is the capital of india?")
print(result.content)