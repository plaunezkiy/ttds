import xml.etree.ElementTree as ET
from utils.xmlparser import XmlDictConfig
from utils.indexing import InvertedFrequencyIndex


index = InvertedFrequencyIndex()

def load_collection(collection_name="sample.xml"):
    # collection = "sample.xml"
    tree = ET.parse(f"data/collections/{collection_name}")
    root = tree.getroot()
    docs = [XmlDictConfig(child) for child in root]

    for doc in docs:
        doc_id = int(doc.get("DOCNO", None))
        doc_text = doc.get("Text", "").strip()
        if not doc_id or not doc_text:
            print(f"Error, empty entry")
            continue
        index.add_document_to_index(doc_id, doc_text)
