import re
from utils.processing import tokenize_text, remove_stopwords, normalize


class InvertedFrequencyIndex:
    """
    Inverted Frequency Index - digests documents and creates a map of 
    processed tokens onto the relevant documents with positions
    """
    # `and:` - extracts 'and' before colon
    term_regex = re.compile(r"(.*?):")
    # `    4: 1,2,3` - splits into groups: (group_1: '4'), (group_2: '1,2,3')
    doc_positions_regex = re.compile("\t(.*?):\s(.*)")
    
    terms: dict
    document_processing_pipeline: list

    def __init__(self):
        self.terms = {}
        self.document_processing_pipeline = [tokenize_text, remove_stopwords, normalize]
    
    def add_document_to_index(self, document_id, text):
        tokens = text
        for preprocessing_func in self.document_processing_pipeline:
            tokens = preprocessing_func(tokens)
        
        for position, token in enumerate(tokens):
            self.add_term_to_index(token, document_id, position+1)
    
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

    def search(self, query):
        return
    
    def save(self, index_path):
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
    
    def load(self, index_path):
        """
        Loads the serialized index line by line (for efficiency)
        """
        with open(index_path, "r") as index_file:
            term = None
            for index_line in index_file:
                # if the only char on the line is a newline - it is a splitter
                if index_line == "\n":
                    # switch to new term
                    term = None
                    continue
                # if the term is not set, we are starting to read the new entry
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
                self.terms[term][doc_id] = positions


if __name__ == "__main__":
    index = InvertedFrequencyIndex()
    index.load("data/collections/collection.index.txt")
