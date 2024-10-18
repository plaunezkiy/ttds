"""
This is ranked retrieval, not Boolean search, 
so you should retrieve all documents that contain 
at least one of the query terms.  

Does this mean we need to perform the 
boolean 'OR' operation first to obtain the list of 
documents containing at least one query term, 
and then calculate the score for each document 
using the provided formula? 

And for each term in the query, 
if the term is not existed in a particular document, 
than the 𝑤(weight) for that term in that document is 0?
"""




def calculate_score(doc, query):
    return
