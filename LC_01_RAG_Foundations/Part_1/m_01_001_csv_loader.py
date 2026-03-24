"""
Script: CSV Loader Example
Description: Demonstrates loading CSV data into LangChain Documents.
"""

from langchain.document_loaders import CSVLoader

def main():
    loader = CSVLoader("./data/raw/data.csv")
    documents = loader.load()

    for doc in documents[:3]:
        print(doc.page_content)
        print(doc.metadata)

if __name__ == "__main__":
    main()