import xml.etree.ElementTree as ET
from utils.xmlparser import Document
from utils.indexing import InvertedFrequencyIndex
import json


index = InvertedFrequencyIndex()

def load_collection(collection_name="trec.sample.xml"):
    # collection = "sample.xml"
    tree = ET.parse(f"data/collections/{collection_name}")
    root = tree.getroot()
    docs = [Document(child) for child in root]
    for doc in docs:
        if not doc.docno or not doc.text:
            print(f"Error, empty entry", doc.docno, doc.text[:10])
            continue
        document_content = doc.headline + doc.text
        index.add_document_to_index(int(doc.docno), document_content)


if __name__ == "__main__":
    load_collection("trec.sample.xml")
    query = input("Q:")
    while True:
        results = index.search(query)
        print(f"({len(results)} matches)")
        print(results)
        query = input("Q:")