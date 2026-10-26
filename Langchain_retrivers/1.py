
from langchain_community.retrievers import WikipediaRetriever

retriever=WikipediaRetriever(top_k_results=2,language='en')
query="What is the significance of space exploration?"
docs=retriever.invoke(query)
for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
    print("\n")
