from langchain_community.document_loaders import CSVLoader
loader=CSVLoader(file_path=r"C:\Users\hp\OneDrive\Desktop\ML\gen ai\document loader\Social_Network_Ads.csv")
data=loader.load()
print(data[0])
print(len(data))
print(data[1].metadata)