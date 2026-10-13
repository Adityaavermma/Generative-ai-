from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
model=ChatOpenAI()
"""messages=[
SystemMessage(content="You are a helpful assistant that summarizes research papers in a concise and clear manner."),
HumanMessage(content="tell me about langchain")
]
result=model.invoke(messages)
messages.append(AIMessage(content=result.content)) 
"""
chat_history=[
    SystemMessage(content="You are a helpful assistant that summarizes research papers in a concise and clear manner."),
]
while True:
    user_input=input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        break
    response=model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("Bot:", response.content)

print(chat_history)