from langchain_community.document_loaders import WebBaseLoader
url="https://www.flipkart.com/apple-macbook-air-m5-2026-m5-16-gb-512-gb-ssd-tahoe-mdhe4hn-a/p/itm8505e2f874525?pid=COMHZQX47WQ7YJZP&lid=LSTCOMHZQX47WQ7YJZPBZOHQJ&marketplace=FLIPKART&store=6bo%2Fb5g&spotlightTagId=default_BestsellerId_6bo%2Fb5g&srno=b_1_1&otracker=browse&fm=organic&iid=7ae2ba15-ceb6-4049-89f3-a91c21fb4ea5.COMHZQX47WQ7YJZP.SEARCH&ppt=sp&ppn=productListView&ssid=726qp0k1ps0000001792367499695&ov_redirect=true"
loader=WebBaseLoader(url)
docs=loader.load()
print(len(docs))
print(docs[0].page_content)

