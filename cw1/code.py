from utils.indexing import InvertedFrequencyIndex


if __name__ == "__main__":
    print("Initalizing the index")
    index = InvertedFrequencyIndex()
    index.load_and_index_collection("cw1/trec.5000.xml")
    
    # generate index.txt
    print("Save the index")
    index.save_to_file("cw1/index.txt")
    
    # generate results.boolean.txt
    print("Running boolean queries")
    with open("cw1/queries.boolean.txt", "r") as q_file:
        with open("cw1/results.boolean.txt", "w") as o_file:
            for query_data in q_file:
                query_data = query_data.strip()
                q_number, query = query_data.split(" ", 1)
                results = index.search(query)
                for doc_id in results:
                    o_file.write(f"{q_number},{doc_id}\n")

    # generate results.ranked.txt
    print("Running ranked queries")
    with open("cw1/queries.ranked.txt", "r") as q_file:
        with open("cw1/results.ranked.txt", "w") as o_file:
            for query_data in q_file:
                query_data = query_data.strip()
                q_number, query = query_data.split(" ", 1)
                results = index.ranked_retrieval(query)
                for (doc_id, tfidf_score) in results:
                    o_file.write(f"{q_number},{doc_id},{tfidf_score}\n")
