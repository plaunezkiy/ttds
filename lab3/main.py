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
    with open("lab3/ranked.queries.txt", "r") as q_file:
        with open("lab3/tfidf.results.txt", "w") as o_file:
            for query_data in q_file:
                query_data = query_data.strip()
                q_number, query = query_data.split(" ", 1)
                results = index.ranked_retrieval(query)
                for (doc_id, tfidf_score) in results:
                    o_file.write(f"{q_number},{doc_id},{tfidf_score}\n")
