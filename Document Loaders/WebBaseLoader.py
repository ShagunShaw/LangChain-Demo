# A WebBaseLoader is a document loader that fetches documents from web pages URLs. It uses two python libraries internally to scrape these web pages: requests and beautifulsoup4.

# NOTE: Now 'WebBaseLoader' works well with static web pages, but it cannot handle dynamic web pages that require JavaScript execution to load content. To address this limitation, we can use 'Selenium' for web scraping. Selenium is a powerful tool that can automate web browsers and interact with dynamic content.


from langchain_community.document_loaders import WebBaseLoader

url= "https://en.wikipedia.org/wiki/Web_scraping"
loader= WebBaseLoader(url)

docs= loader.load()

print(len(docs))  # Will print 1, as there is one document loaded from the each URL

print(docs[0].page_content)  # Print the entire content of the web page


# Now, we can also load multiple URLs at once by passing a list of URLs to the WebBaseLoader.

urls= ["https://en.wikipedia.org/wiki/Web_scraping",
       "https://en.wikipedia.org/wiki/Data_mining", "https://en.wikipedia.org/wiki/Machine_learning"]

loader2= WebBaseLoader(urls)

docs2= loader2.load()

print(len(docs2))  # Will print 3, as there are three documents loaded from the three URLs