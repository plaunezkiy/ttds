import re
from math import log10
from typing import Set
import xml.etree.ElementTree as ET
from utils.processing import tokenize_text, remove_stopwords, normalize
from utils.xmlparser import Document

"""
TODO: think about a more efficient way of storing and loading
the serialized index. Currently reading line by line from a txt file
Supposedely efficient in terms of memory for file IO

TODO: Implement delta encoding for more memory efficient term-in-doc indexing
"""

class InvertedFrequencyIndex:
    """
    Inverted Frequency Index - digests documents and creates a map of 
    processed tokens onto the relevant documents with positions
    """
    # `and:` - extracts 'and' before colon
    term_regex = re.compile(r"(.*?):")
    # `    4: 1,2,3` - splits into groups: (group_1: '4'), (group_2: '1,2,3')
    doc_positions_regex = re.compile("\t(.*?):\s(.*)")
    
    doc_ids: set
    terms: dict
    text_processing_pipeline: list

    def __init__(self):
        self.doc_ids = set()
        self.terms = {}
        self.text_processing_pipeline = [tokenize_text, remove_stopwords, normalize]
    
    def add_document_to_index(self, document_id, text):
        tokens = text
        for preprocessing_func in self.text_processing_pipeline:
            tokens = preprocessing_func(tokens)
        for position, token in enumerate(tokens):
            self.add_term_to_index(token, document_id, position+1)
        self.doc_ids.add(document_id)
    
    def add_term_to_index(self, term: str, document_id: int, position: int):
        """
        Takes a term, doc_id, position in the doc
        and adds it to the index map
        """
        if term in self.terms:
            if document_id in self.terms[term]:
                self.terms[term][document_id].append(position)
            else:
                self.terms[term][document_id] = [position]
        else:
            self.terms[term] = {
                document_id: [position]
            }
    
    def check_term_in_document(self, doc_id, term) -> bool:
        """
        check if the term is present in a given document
        """
        return True if self.terms[term][doc_id] else False

    def term_search(self, term):
        """
        Returns list of documents containing the term, otherwise an empty list
        `term` to be in processed and normalized form
        """
        keys = self.terms.get(term, {}).keys()
        return set(keys)
    
    def phrase_search(self, query, exact=False) -> Set[int]:
        """
        Finds documents that contain a string of text
        Splits the query into tokens and does proximity search over
        each adjacent pair of tokens, if exact, only documents where pairs
        appear sequentially will be returned.
        Otherwise, all document with a pair occurrence are union'ed
        """
        # if exact (exact match), enforce set intersection, 
        # else union over documents containing pairs
        set_operator = set.intersection if exact else set.union
        # converty query into tokens and process them
        query_tokens = query
        for func in self.text_processing_pipeline:
            query_tokens = func(query_tokens)
        # if no tokens (either all stopwords or or nothing provided, return)
        if len(query_tokens) == 0:
            return set()
        # if 1 token left, can do simple term search
        if len(query_tokens) == 1:
            return self.term_search(query_tokens[0])
        # otherwise, zip pairs and do proximity search for each pair
        docs = set()
        for pair in zip(query_tokens, query_tokens[1:]):
            result = self.proximity_search(*pair, n=1)
            # if not the first pass, apply operator
            # avoids difference of an empty set() and nonempty on the first pass
            if docs:
                docs = set_operator(docs, result)
            else:
                docs = result
        return docs
    
    def check_terms_close_in_document(self, doc_id, term1, term2, n) -> bool:
        # term occurrences in a document
        i1 = self.terms.get(term1, {}).get(doc_id, [])
        i2 = self.terms.get(term2, {}).get(doc_id, [])
        # list pointers
        i = 0
        j = 0
        while j < len(i2):
            # while i1 is before i2 and while the diff > n, move to next i1
            while i < len(i1)-1 and i1[i] <= i2[j] and abs(i2[j] - i1[i]) > n:
                i += 1
            if abs(i2[j] - i1[i]) <= n:
                return True
            j += 1
        return False

    def proximity_search(self, term1, term2, n) -> Set[int]:
        """
        Performs proximity search over the index
        Returns relevant documents, where both term1 and term2
        are present, and the distance between them is <= n

        term1 and term2 are to be in a processed normalized form
        """
        # position lists
        d1 = self.terms.get(term1, {})
        d2 = self.terms.get(term2, {})
        relevant_doc_ids = set()
        for doc_id in d1.keys():
            # if doc_id not common, continue
            if doc_id not in d2:
                continue
            if self.check_terms_close_in_document(doc_id, term1, term2, n):
                relevant_doc_ids.add(doc_id)
        return relevant_doc_ids
    
    def evaluate_expression(self, exp):
        negated = False
        docs = set()
        if "NOT" in exp:
            negated = True
            exp = re.findall(r"NOT\s?(.*)", exp)[0]
        # #int(str,str)
        proximity_regex = r"#(\d+)\((\w+),\s?(\w+)\)"
        # "str"
        phrase_regex = r"\"(.*)\""
        if re.search(proximity_regex, exp):
            n, t1, t2 = re.findall(proximity_regex, exp)[0]
            tokens = [t1, t2]
            # skip tokenization
            for preproc_func in self.text_processing_pipeline[1:]:
                tokens = preproc_func(tokens)
            docs = self.proximity_search(*tokens, int(n))
        # phrase search
        elif re.search(phrase_regex, exp):
            exact_query = re.findall(phrase_regex, exp)[0]
            docs = self.phrase_search(exact_query, exact=True)
        # regular search
        else:
            docs = self.phrase_search(exp)
        
        if negated:
            return self.doc_ids.difference(docs)
        return docs

    def search(self, query):
        # Boolean search (AND, OR, NOT)
        # phrase search ("exact match")
        # proximity search pos(term2) – pos(term1) < |w| -> #5(term1,term2)
        # ranked IR search based on TFIDF
        operator = None
        # split on AND/OR, if present, otherwise just find
        # Left AND/OR Right
        if "AND" in query:
            operator = set.intersection
            query = query.split("AND")
        elif "OR" in query:
            operator = set.union
            query = query.split("OR")
        else:
            return sorted(self.evaluate_expression(query))
        return sorted(operator(*map(self.evaluate_expression, query)))

    def get_tf(self, term, doc_id):
        """
        Calculates the term frequency in a document
        ie how many times term appeared in a document
        """
        fs = self.terms.get(term, {}).get(doc_id, [])
        return len(fs)
    
    def get_df(self, term):
        """
        Calculates how many documents the term appeared in.
        """
        docs = self.terms.get(term, {})
        return len(docs.keys())
    
    def ranked_retrieval(self, query):
        # preprocess the query
        query_tokens = query
        for preproc_func in self.text_processing_pipeline:
            query_tokens = preproc_func(query_tokens)
        # 
        docs = set.union(*[self.term_search(term) for term in query_tokens])
        ranked_docs = []
        # 
        for doc_id in docs:
            doc_score = 0
            for term in query_tokens:
                # term-doc frequency
                tf = self.get_tf(term, doc_id)
                if tf == 0:
                    w = 0
                else:
                    # idf over entire collection
                    idf = log10(len(self.doc_ids) / self.get_df(term))
                    w = (1 + log10(tf)) * idf
                doc_score += w
            ranked_docs.append((doc_id, doc_score))
        # 
        return ranked_docs
        # sorted
        return sorted(ranked_docs, key=lambda t: t[1], reverse=True)

    def load_and_index_collection(self, collection_path):
        """
        Loads and indexes a collection of documents from XML file
        """
        tree = ET.parse(collection_path)
        root = tree.getroot()
        docs = [Document(child) for child in root]
        for doc in docs:
            if not doc.docno or not doc.text:
                print(f"Error, missing data for the entry", doc.docno, doc.text[:10])
                continue
            document_content = doc.headline + doc.text
            self.add_document_to_index(int(doc.docno), document_content)

    def save_to_file(self, index_path):
        """
        Stores the index in the serializable format
        
        ```
        token:
            4: 1,2,3
        ```
        """
        with open(index_path, "w") as index_file:
            for term in self.terms:
                doc_freqs = "\n".join(
                    [
                        f"\t{doc_id}: {','.join(str(pos) for pos in self.terms[term][doc_id])}" for doc_id in self.terms[term]
                    ]
                )
                term_index_data = f"{term}:\n{doc_freqs}\n\n"
                index_file.write(term_index_data)
    
    def load_from_file(self, index_path):
        """
        Loads from the serialized index file line by line (for efficiency)
        """
        with open(index_path, "r") as index_file:
            term = None
            for index_line in index_file:
                # if the only char on the line is a newline - it is a splitter
                if index_line == "\n":
                    # switch to new term
                    term = None
                    continue
                # if the term is not set, we are starting the new entry
                # and thus it needs to be set
                if term == None:
                    result = self.term_regex.search(index_line)
                    term = result.group(1)
                    self.terms[term] = {}
                    continue
                # otherwise, it is a document_id with positions
                res = self.doc_positions_regex.search(index_line)
                doc_id = int(res.group(1))
                positions = map(int, res.group(2).split(","))
                self.doc_ids.add(doc_id)
                self.terms[term][doc_id] = positions


if __name__ == "__main__":
    index = InvertedFrequencyIndex()
    index.load_from_file("data/collections/collection.index.txt")
    query = input("Q:")
    while True:
        results = index.ranked_retrieval(query)
        print(f"({len(results)} matches)")
        print(results, sep="\n")
        query = input("Q:")
